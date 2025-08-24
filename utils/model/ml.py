
from typing import Dict, List
import pandas as pd
from xgboost import XGBRegressor

STAGE1_TARGETS = {
    'Nacelle\nWind Speed\n[m/s]': 'pred_Nacelle_Wind_Speed',
    'Nacelle\nWind Direction\n[deg]': 'pred_Nacelle_Wind_Direction',
    'Nacelle\nOutdoor Temp\n[℃]': 'pred_Nacelle_Outdoor_Temp',
}

def fit_stage1_models(train_df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, XGBRegressor]:
    models = {}
    for tgt in STAGE1_TARGETS.keys():
        if tgt not in train_df.columns:
            continue
        m = XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.9, random_state=42, n_jobs=-1
        )
        m.fit(train_df[feature_cols], train_df[tgt])
        models[tgt] = m
    return models

def infer_stage1(models: Dict[str, XGBRegressor], df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for tgt, out_name in STAGE1_TARGETS.items():
        m = models.get(tgt, None)
        if m is None:
            continue
        out[out_name] = m.predict(df[feature_cols])
    return out

def fit_stage2_power(X, y) -> XGBRegressor:
    m = XGBRegressor(
        n_estimators=800, max_depth=8, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.9, random_state=42, n_jobs=-1
    )
    m.fit(X, y)
    return m
