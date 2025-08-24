
import numpy as np
import pandas as pd
from typing import List

def add_basic_time_features(df: pd.DataFrame, time_col='Date/Time') -> pd.DataFrame:
    ts = pd.to_datetime(df[time_col])
    df['hour'] = ts.dt.hour
    df['day'] = ts.dt.day
    df['month'] = ts.dt.month
    df['day_of_week'] = ts.dt.dayofweek
    return df

def add_diff_features(df: pd.DataFrame, cols: List[str], group_cols: List[str] = None, suffix='_diff'):
    """Add first differences for selected columns. If group_cols is given, compute diffs per group."""
    d = df.copy()
    if group_cols is None:
        for c in cols:
            if c in d.columns:
                d[c + suffix] = d[c].diff()
    else:
        order_cols = group_cols + (['Date/Time'] if 'Date/Time' in d.columns else [])
        d = d.sort_values(order_cols)
        for c in cols:
            if c in d.columns:
                d[c + suffix] = d.groupby(group_cols)[c].diff()
    return d

def wind_speed_binning(df: pd.DataFrame, col='vws_10m'):
    if col not in df.columns:
        return df
    bins = [0, 3, 6, 9, 12, 15, 20, 30]
    labels = ['0-3','3-6','6-9','9-12','12-15','15-20','20+']
    df['wind_speed_bin'] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    dummies = pd.get_dummies(df['wind_speed_bin'], prefix='wind_bin')
    return pd.concat([df, dummies], axis=1)

def make_stage2_features(df: pd.DataFrame):
    """Stage-2 uses forecast features + Stage-1 predicted (pseudo-observed) nacelle signals."""
    feature_cols = [
        'vws_10m','uws_10m','ta','rh_1p5m','pmsl','dswrf',
        'hour','day_of_week',
        'pred_Nacelle_Wind_Speed', 'pred_Nacelle_Wind_Direction', 'pred_Nacelle_Outdoor_Temp'
    ]
    exist = [c for c in feature_cols if c in df.columns]
    return df[exist].copy()
