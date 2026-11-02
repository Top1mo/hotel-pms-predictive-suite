import pandas as pd

def clean_dataset(df):
    """
    Sanitizes the raw H2 dataset by removing duplicates, handling extreme outliers, 
    and imputing missing values.
    
    Args:
        df (pd.DataFrame): Raw dataframe loaded directly from H2.csv.
        
    Returns:
        pd.DataFrame: Cleaned dataframe ready for feature engineering.
    """
    df = df.copy()

    # 1. Deduplication
    # Removes exact duplicate rows inherent to group bookings in the PMS
    df = df.drop_duplicates().reset_index(drop=True)

    # 2. Outlier Removal
    # Drops the single massive ADR anomaly (5400) to prevent regression distortion
    if 'ADR' in df.columns:
        df = df[df['ADR'] < 1000]

    # 3. Standard Missing Value Imputation
    if 'Children' in df.columns:
        df['Children'] = df['Children'].fillna(0)
        
    if 'Country' in df.columns:
        df['Country'] = df['Country'].fillna('PRT')

    # 4. Hidden Missing Value Imputation ('NULL' strings)
    if 'Agent' in df.columns:
        df['Agent'] = df['Agent'].astype(str).str.strip()
        df['Agent'] = df['Agent'].replace('NULL', 'No_Agent')
        
    if 'Company' in df.columns:
        # Standardize the string; the actual flag conversion happens in build_features.py
        df['Company'] = df['Company'].astype(str).str.strip()

    return df

if __name__ == "__main__":
    # Example usage for testing the module independently
    # raw_data = pd.read_csv('../../data/H2.csv')
    # clean_data = clean_dataset(raw_data)
    # print(clean_data.isnull().sum())
    pass