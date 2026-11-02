import pandas as pd
import numpy as np

def engineer_features(df):
    """
    Applies feature engineering to the cleaned H2 dataset for Milestone 1.
    Handles dates, creates interaction features, and removes leakage.
    
    Args:
        df (pd.DataFrame): Cleaned dataframe.
        
    Returns:
        pd.DataFrame: Processed dataframe ready for pipeline encoding/scaling.
    """
    df = df.copy()

    # 1. Leakage Removal (Core Lifecycle Leaks)
    leakage_columns = [
        'ReservationStatus', 
        'ReservationStatusDate', 
        'BookingChanges', 
        'AssignedRoomType'
    ]
    df = df.drop(columns=leakage_columns, errors='ignore')

    # 2. Company Flag
    # Replaces the sparse ID column with a binary flag
    if 'Company' in df.columns:
        df['Has_Company'] = np.where(df['Company'] != 'NULL', 1, 0)
        df = df.drop(columns=['Company'])

    # 3. Standard Date Engineering (Option 1B)
    if 'ArrivalDateMonth' in df.columns:
        month_map = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4,
            'May': 5, 'June': 6, 'July': 7, 'August': 8,
            'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        df['Arrival_Month_Num'] = df['ArrivalDateMonth'].map(month_map)
        
        # Create temporary datetime object to extract weekday
        temp_date = pd.to_datetime({
            'year': df['ArrivalDateYear'],
            'month': df['Arrival_Month_Num'],
            'day': df['ArrivalDateDayOfMonth']
        })
        
        df['ArrivalDayOfWeek'] = temp_date.dt.dayofweek
        df['Is_Weekend_Arrival'] = np.where(df['ArrivalDayOfWeek'] >= 5, 1, 0)
        
        # Drop original raw temporal components
        cols_to_drop = ['ArrivalDateYear', 'ArrivalDateMonth', 'ArrivalDateDayOfMonth', 'ArrivalDateWeekNumber']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # 4. Engineered Combinations (Option 4A)
    if 'StaysInWeekendNights' in df.columns and 'StaysInWeekNights' in df.columns:
        df['Total_Stay_Nights'] = df['StaysInWeekendNights'] + df['StaysInWeekNights']
        
    if all(c in df.columns for c in ['Adults', 'Children', 'Babies']):
        df['Total_Guests'] = df['Adults'] + df['Children'] + df['Babies']
        df['Is_Family'] = np.where((df['Children'] > 0) | (df['Babies'] > 0), 1, 0)

    # Note: Country and Agent are intentionally left as raw strings here (Options 2A & 3A).
    # They should be transformed using TargetEncoder or OneHotEncoder inside 
    # your scikit-learn ColumnTransformer during the modeling phase.

    return df

if __name__ == "__main__":
    # Example usage for testing the module independently
    # df_clean = pd.read_csv('../../data/processed/H2_cleaned_only.csv')
    # df_processed = engineer_features(df_clean)
    # print(df_processed.shape)
    pass