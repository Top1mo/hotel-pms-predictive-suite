"""Load H2 with a verified schema and normalized missing-value tokens."""
from pathlib import Path
import pandas as pd
EXPECTED_COLUMNS = """IsCanceled LeadTime ArrivalDateYear ArrivalDateMonth ArrivalDateWeekNumber ArrivalDateDayOfMonth StaysInWeekendNights StaysInWeekNights Adults Children Babies Meal Country MarketSegment DistributionChannel IsRepeatedGuest PreviousCancellations PreviousBookingsNotCanceled ReservedRoomType AssignedRoomType BookingChanges DepositType Agent Company DaysInWaitingList CustomerType ADR RequiredCarParkingSpaces TotalOfSpecialRequests ReservationStatus ReservationStatusDate""".split()
CATEGORICAL_COLUMNS = """ArrivalDateMonth Meal Country MarketSegment DistributionChannel ReservedRoomType AssignedRoomType DepositType Agent Company CustomerType ReservationStatus ReservationStatusDate""".split()
def load_h2(path, expected_rows=79330):
    df = pd.read_csv(Path(path), dtype={c: 'string' for c in CATEGORICAL_COLUMNS})
    if set(df.columns) != set(EXPECTED_COLUMNS):
        raise ValueError('H2 column names do not match the source schema.')
    if expected_rows is not None and len(df) != expected_rows:
        raise ValueError(f'Expected {expected_rows} rows, received {len(df)}')
    for c in CATEGORICAL_COLUMNS:
        df[c] = df[c].str.strip().replace({'NULL': pd.NA, '': pd.NA, 'Undefined': pd.NA})
    for c in set(EXPECTED_COLUMNS) - set(CATEGORICAL_COLUMNS):
        df[c] = pd.to_numeric(df[c], errors='raise')
    if df[['ADR', 'IsCanceled']].isna().any().any():
        raise ValueError('Missing targets need an explicit task-specific policy.')
    if not df['IsCanceled'].isin([0, 1]).all():
        raise ValueError('IsCanceled must be 0 or 1.')
    df.index.name = 'row_id'
    return df[EXPECTED_COLUMNS]
