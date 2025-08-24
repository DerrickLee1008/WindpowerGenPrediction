
import os
import pickle
import pandas as pd
from typing import List

from ..data.load_data import load_weather_csvs
from ..data.transforms import make_stage2_features
from ..model.ml import infer_stage1

class WindPowerPredictor:
    """
    Inference-only class that consumes *forecast only* and outputs power predictions.
    """
    def __init__(self, artifact_dir: str):
        self.artifact_dir = artifact_dir
        self.stage1_models = None
        self.power_model = None

    def load(self):
        with open(os.path.join(self.artifact_dir, 'stage1_models.pkl'), 'rb') as f:
            self.stage1_models = pickle.load(f)
        with open(os.path.join(self.artifact_dir, 'power_model.pkl'), 'rb') as f:
            self.power_model = pickle.load(f)

    def predict(self, weather_files: List[str]) -> pd.DataFrame:
        assert self.stage1_models is not None and self.power_model is not None, "Call .load() before .predict()"
        wx = load_weather_csvs(weather_files)
        stage1_feats = [c for c in ['vws_10m','uws_10m','ta','rh_1p5m','pmsl','dswrf','hour','day_of_week'] if c in wx.columns]
        wx = infer_stage1(self.stage1_models, wx, stage1_feats)
        X = make_stage2_features(wx)
        power_pred = self.power_model.predict(X)
        out = wx[['Date/Time','WTG']].copy()
        out['pred_power_kw'] = power_pred
        return out

def predict_from_forecasts(weather_files: List[str], artifact_dir: str) -> pd.DataFrame:
    predictor = WindPowerPredictor(artifact_dir=artifact_dir)
    predictor.load()
    return predictor.predict(weather_files)
