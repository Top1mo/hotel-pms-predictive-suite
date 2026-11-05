"""Task-specific pipelines; fit within each cross-validation training fold."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from src.features.build_features import BookingFeatures
SOURCE_FEATURES = """LeadTime ArrivalDateYear ArrivalDateMonth ArrivalDateDayOfMonth StaysInWeekendNights StaysInWeekNights Adults Children Babies Meal Country MarketSegment DistributionChannel IsRepeatedGuest PreviousCancellations PreviousBookingsNotCanceled ReservedRoomType Agent Company CustomerType""".split()
NUMERIC = """LeadTime ArrivalDateYear StaysInWeekendNights StaysInWeekNights Adults Children Babies IsRepeatedGuest PreviousCancellations PreviousBookingsNotCanceled Has_Company Arrival_Month_Sin Arrival_Month_Cos Is_Weekend_Arrival Total_Stay_Nights Total_Guests Is_Family""".split()
CATEGORICAL = """Meal Country MarketSegment DistributionChannel ReservedRoomType Agent CustomerType ArrivalDayOfWeek""".split()
CLUSTER_NUMERIC = 'LeadTime Total_Stay_Nights Total_Guests Has_Company IsRepeatedGuest Arrival_Month_Sin Arrival_Month_Cos'.split()
CLUSTER_CATEGORICAL = ['Meal','MarketSegment','CustomerType','ReservedRoomType']
TARGETS = {'regression':'ADR', 'classification':'IsCanceled'}

def task_data(frame, task):
    if task not in ('regression','classification','clustering'):
        raise ValueError(f'Unknown task: {task}')
    X = frame.loc[:, SOURCE_FEATURES].copy()
    for c in X.select_dtypes(include=['object','string']).columns:
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    return X, frame[TARGETS[task]].copy() if task in TARGETS else None

def category_strings(frame):
    return frame.astype('string').astype(object).where(frame.notna(), np.nan)

def make_preprocessor(task, scale=True):
    if task not in ('regression','classification','clustering'):
        raise ValueError(f'Unknown task: {task}')
    nums = CLUSTER_NUMERIC if task=='clustering' else NUMERIC
    cats = CLUSTER_CATEGORICAL if task=='clustering' else CATEGORICAL
    steps = [('impute', SimpleImputer(strategy='median', keep_empty_features=True))]
    if scale:
        steps.append(('scale', StandardScaler()))
    categorical = Pipeline([
        ('strings', FunctionTransformer(category_strings, feature_names_out='one-to-one')),
        ('impute', SimpleImputer(strategy='constant', fill_value='Missing', keep_empty_features=True)),
        ('encode', OneHotEncoder(handle_unknown='ignore', min_frequency=5, max_categories=50, sparse_output=True))])
    columns = ColumnTransformer([('numeric', Pipeline(steps), nums), ('categorical', categorical, cats)], remainder='drop', sparse_threshold=1.0)
    return Pipeline([('features', BookingFeatures()), ('columns', columns)])

def make_model_pipeline(task, estimator, scale=True):
    return Pipeline([('preprocess', make_preprocessor(task, scale)), ('model', estimator)])
