"""Row-local features; no fitted statistics or target information."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
MONTHS = {m: i+1 for i,m in enumerate(['January','February','March','April','May','June','July','August','September','October','November','December'])}

def engineer_features(df):
    df = df.copy()
    company = df['Company'].astype('string').str.strip().replace({'NULL': pd.NA, '': pd.NA})
    df['Has_Company'] = company.notna().astype(int)
    month = df['ArrivalDateMonth'].map(MONTHS).astype(float)
    date = pd.to_datetime(dict(year=df['ArrivalDateYear'], month=month, day=df['ArrivalDateDayOfMonth']), errors='raise')
    df['Arrival_Month_Sin'] = np.sin(2*np.pi*month/12)
    df['Arrival_Month_Cos'] = np.cos(2*np.pi*month/12)
    df['ArrivalDayOfWeek'] = date.dt.dayofweek.astype(float)
    df['Is_Weekend_Arrival'] = (date.dt.dayofweek >= 5).astype(int)
    df['Total_Stay_Nights'] = df['StaysInWeekendNights'] + df['StaysInWeekNights']
    df['Total_Guests'] = df['Adults'] + df['Children'] + df['Babies']
    df['Is_Family'] = np.where((df['Children']>0)|(df['Babies']>0), 1., np.where(df[['Children','Babies']].isna().any(axis=1), np.nan, 0.))
    return df

class BookingFeatures(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return engineer_features(X)
