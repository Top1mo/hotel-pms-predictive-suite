"""Group-held-out partitions prevent identical booking inputs crossing splits."""
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from src.preprocessing.pipelines import SOURCE_FEATURES

def booking_groups(frame):
    # No outcomes in grouping: identical inputs with different labels still stay together.
    return pd.util.hash_pandas_object(frame[SOURCE_FEATURES], index=False).astype(str)

def split_bookings(frame, seed=42, test_size=.2, val_size=.2):
    if not (0 < test_size < 1 and 0 < val_size < 1 and test_size+val_size < 1):
        raise ValueError('Validation and test fractions must be positive and sum to less than 1.')
    groups = booking_groups(frame)
    first = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    rest, test = next(first.split(frame, groups=groups))
    second = GroupShuffleSplit(n_splits=1, test_size=val_size/(1-test_size), random_state=seed)
    tr, va = next(second.split(frame.iloc[rest], groups=groups.iloc[rest]))
    positions = {'train':rest[tr], 'validation':rest[va], 'test':test}
    manifest = pd.DataFrame({'row_id':frame.index, 'group_id':groups.to_numpy(), 'split':''})
    for name,pos in positions.items():
        manifest.loc[pos, 'split'] = name
    assert manifest.groupby('group_id')['split'].nunique().max() == 1
    return {name:frame.iloc[pos].copy() for name,pos in positions.items()}, manifest

def training_cv(frame, n_splits=5):
    """Use returned cv and groups with the model pipeline on TRAINING rows only."""
    return GroupKFold(n_splits=n_splits), booking_groups(frame)
