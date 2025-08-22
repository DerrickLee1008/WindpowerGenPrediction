
import os
import pickle
import pandas as pd
from typing import List
from ..data.load_data import load_weather_csvs
from ..data.transforms import make_stage2_features
from ..model.ml import infer_stage1

def load_models(artifact_dir: str):
    with open(os.path.join(artifact_dir, 'stage1_models.pkl'), 'rb') as f:
        s1 = pickle.load(f)
    with open(os.path.join(artifact_dir, 'power_model.pkl'), 'rb') as f:
        p = pickle.load(f)
    return s1, p

def predict_from_forecasts(
    weather_files: List[str],
    artifact_dir: str
) -> pd.DataFrame:
    """
    Inference-only path that consumes *forecast only* (no SCADA) and outputs power predictions.
    Assumes weather_files already represent the valid issuance rule (D-1 15:00) upstream.
    """
    wx = load_weather_csvs(weather_files)
    s1, pwr = load_models(artifact_dir)
    stage1_feats = [c for c in ['vws_10m','uws_10m','ta','rh_1p5m','pmsl','dswrf','hour','day_of_week'] if c in wx.columns]

    wx = infer_stage1(s1, wx, stage1_feats)
    X = make_stage2_features(wx)
    power_pred = pwr.predict(X)

    out = wx[['Date/Time','WTG']].copy()
    out['pred_power_kw'] = power_pred
    return out
