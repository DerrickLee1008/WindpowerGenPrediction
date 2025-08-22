
import os
import re
import pandas as pd
from typing import List, Optional
from .transforms import add_basic_time_features
from ..common.utils import safe_float_convert

SCADA_NUMERIC_COLS = [
    'Grid\nActive Power\n[kW]',
    'Nacelle\nWind Speed\n[m/s]',
    'Nacelle\nWind Direction\n[deg]',
    'Nacelle\nOutdoor Temp\n[℃]',
    'Rotor\nRotor Speed\n[rpm]'
]

def _wtg_from_filename(path: str) -> Optional[str]:
    m = re.search(r'(WTG\d{2})', os.path.basename(path))
    return m.group(1) if m else None

def load_weather_csvs(weather_files: List[str]) -> pd.DataFrame:
    dfs = []
    for fp in weather_files:
        df = pd.read_csv(fp)
        if 'forecast_target_time' in df.columns:
            df['forecast_target_time'] = pd.to_datetime(df['forecast_target_time'])
        wtg = _wtg_from_filename(fp) or 'WTG01'
        df['WTG'] = wtg
        dfs.append(df)
    wx = pd.concat(dfs, ignore_index=True).sort_values(['WTG','forecast_target_time'])
    wx.rename(columns={'forecast_target_time': 'Date/Time'}, inplace=True)
    wx = add_basic_time_features(wx, time_col='Date/Time')
    return wx

def load_scada_excel(scada_path: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    df = pd.read_excel(scada_path, skiprows=4)
    # First row is header repeated by the exporter
    df.columns = df.iloc[0]
    df = df.iloc[1:].reset_index(drop=True)
    # Drop last 3 summary rows if present
    if len(df) >= 3:
        df = df.iloc[:-3].reset_index(drop=True)
    # Standardize
    if 'Date/Time' in df.columns:
        df['Date/Time'] = pd.to_datetime(df['Date/Time'])
    safe_float_convert(df, SCADA_NUMERIC_COLS)
    # Keep basic columns
    keep_cols = ['Date/Time', 'WTG. Name'] + SCADA_NUMERIC_COLS
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()
    df.rename(columns={'WTG. Name':'WTG'}, inplace=True)
    # Filter by date window if given
    if start_date:
        df = df[df['Date/Time'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['Date/Time'] <= pd.to_datetime(end_date)]
    df.sort_values(['WTG','Date/Time'], inplace=True)
    return df

def merge_weather_scada(weather_df: pd.DataFrame, scada_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge on Date/Time AND WTG to avoid cross-turbine leakage.
    """
    merged = pd.merge(scada_df, weather_df, on=['Date/Time','WTG'], how='inner')
    return merged
