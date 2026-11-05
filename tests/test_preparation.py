import io
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from src.data.loader import load_h2
from src.data.prepare import ROOT
from src.data.splits import split_bookings, booking_groups, training_cv
from src.preprocessing.cleaning import clean_dataset
from src.preprocessing.pipelines import task_data, make_preprocessor
from src.features.build_features import engineer_features

@pytest.fixture
def sample():
    return load_h2(ROOT/'data/raw/H2.csv').iloc[:250].copy()

def test_schema_rejects_wrong_columns(tmp_path, sample):
    p=tmp_path/'bad.csv'
    sample.drop(columns='ADR').to_csv(p,index=False)
    with pytest.raises(ValueError,match='schema'):
        load_h2(p, expected_rows=None)

def test_cleaning_removes_duplicates_but_retains_extremes_and_unknowns(sample):
    sample=sample.iloc[:3].copy()
    sample.loc[sample.index[0], ['Adults','Children','Babies']]=[0,0,0]
    sample.loc[sample.index[1], ['Adults','Children','Babies']]=[0,np.nan,0]
    sample.loc[sample.index[2], 'ADR']=5400
    doubled=pd.concat([sample,sample.iloc[[2]]])
    result=clean_dataset(doubled)
    assert len(result)==2
    assert not result.duplicated().any()
    assert result['ADR'].max()==5400
    assert result['Children'].isna().sum()==1

def test_company_missing_variants(sample):
    sample=sample.iloc[:5].copy()
    sample['Company']=[None,np.nan,' NULL ','','40']
    assert engineer_features(sample)['Has_Company'].tolist()==[0,0,0,0,1]

def test_split_reproducible_disjoint_complete_and_outcome_independent(sample):
    sample=clean_dataset(sample).reset_index(drop=True)
    extra=sample.iloc[[0]].copy()
    extra['IsCanceled']=1-extra['IsCanceled'];extra['ADR']=9999
    sample=pd.concat([sample,extra],ignore_index=True)
    a,m=split_bookings(sample); b,n=split_bookings(sample)
    pd.testing.assert_frame_equal(m,n)
    assert sorted(m.row_id)==list(sample.index)
    assert m.groupby('group_id')['split'].nunique().max()==1
    assert m.iloc[0]['split']==m.iloc[-1]['split']
    for name in a: pd.testing.assert_frame_equal(a[name],b[name])
    cv,groups=training_cv(a['train'],3)
    for tr,va in cv.split(a['train'],groups=groups):
        assert set(groups.iloc[tr]).isdisjoint(groups.iloc[va])

@pytest.mark.parametrize('task',['regression','classification','clustering'])
def test_train_only_statistics_unknown_categories_and_serialization(sample,task):
    train=clean_dataset(sample.iloc[:150]).copy()
    val=clean_dataset(sample.iloc[150:]).copy()
    train['LeadTime']=10.;train.iloc[0,train.columns.get_loc('LeadTime')]=np.nan
    val['LeadTime']=1000000.;val['Meal']='NEVER_SEEN'
    X,y=task_data(train,task);V,_=task_data(val,task)
    assert not {'ADR','IsCanceled','ReservationStatus','BookingChanges'} & set(X.columns)
    p=clone(make_preprocessor(task))
    A=p.fit_transform(X,y)
    numeric=p.named_steps['columns'].named_transformers_['numeric']
    assert numeric.named_steps['impute'].statistics_[0]==10.
    before=numeric.named_steps['impute'].statistics_.copy()
    B=p.transform(V)
    assert A.shape[1]==B.shape[1]
    assert np.isfinite(B.data).all()
    np.testing.assert_array_equal(before,numeric.named_steps['impute'].statistics_)
    # Supervised targets do not influence any fitted preprocessing statistics.
    other=clone(make_preprocessor(task)).fit_transform(X, None if y is None else y.iloc[::-1])
    np.testing.assert_allclose(A.toarray(),other.toarray())
    buffer=io.BytesIO();joblib.dump(p,buffer);buffer.seek(0)
    np.testing.assert_allclose(joblib.load(buffer).transform(V).toarray(),B.toarray())


def test_cleaning_keeps_distinct_outcomes_and_reports_counts(sample):
    from src.preprocessing.cleaning import cleaning_report
    first = sample.iloc[[0]].copy()
    first['Adults'] = 2
    other = first.copy()
    other['ADR'] = first['ADR'] + 10
    other['IsCanceled'] = 1 - first['IsCanceled']
    raw = pd.concat([first, first.copy(), other], ignore_index=True)
    clean = clean_dataset(raw)
    assert len(clean) == 2
    assert clean.index.tolist() == [0, 2]
    report = cleaning_report(raw, clean)
    assert report['removed_exact_duplicate_rows'] == 1
    assert report['removed_known_zero_guest_rows'] == 0
    assert report['exact_duplicates_remaining'] == 0
