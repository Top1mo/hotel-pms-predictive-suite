"""Deterministic cleaning; statistical imputation is fitted in training pipelines."""
import pandas as pd
from src.data.loader import CATEGORICAL_COLUMNS


def normalize_categories(df):
    df = df.copy()
    for column in CATEGORICAL_COLUMNS:
        if column in df:
            df[column] = df[column].astype('string').str.strip().replace(
                {'NULL': pd.NA, '': pd.NA, 'Undefined': pd.NA})
    return df


def eligible_rows(df):
    # Unknown child count is not proof that all guest counts are zero.
    zero_guests = df[['Adults', 'Children', 'Babies']].eq(0).fillna(False).all(axis=1)
    return df.loc[~zero_guests].copy()


def clean_dataset(df):
    df = eligible_rows(normalize_categories(df))
    # Full-row duplicates only; distinct outcomes must not be discarded.
    # Preserve the first original row ID for the split audit manifest.
    return df.drop_duplicates(keep='first').copy()


def cleaning_report(raw, clean):
    normalized = normalize_categories(raw)
    eligible = eligible_rows(normalized)
    return {
        'input_rows': len(raw),
        'removed_known_zero_guest_rows': len(normalized) - len(eligible),
        'removed_exact_duplicate_rows': int(eligible.duplicated().sum()),
        'retained_rows': len(clean),
        'exact_duplicates_remaining': int(clean.duplicated().sum()),
        'policy': 'Normalize categories; remove known zero-guest rows; remove exact full-row '
                  'duplicates before splitting. Retain ADR extremes and missing values. '
                  'Group identical booking inputs across splits even when outcomes differ.'
    }
