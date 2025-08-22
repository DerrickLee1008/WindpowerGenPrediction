
import os
import pickle
import pandas as pd
from typing import Dict, Tuple, List
from ..data.load_data import load_weather_csvs, load_scada_excel, merge_weather_scada
from ..data.transforms import add_diff_features, wind_speed_binning, make_stage2_features, add_basic_time_features
from ..model.ml import fit_stage1_models, infer_stage1, fit_stage2_power
from ..common.utils import ensure_dir, set_seed
from ..common.loss_function import mae, rmse, r2, safe_mape, nmae

TARGET_COL = 'Grid\nActive Power\n[kW]'

def time_order_split(df: pd.DataFrame, ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    idx = int(n * ratio)
    return df.iloc[:idx].copy(), df.iloc[idx:].copy()

def train_pipeline(
    weather_files: List[str],
    scada_path: str,
    start_date: str,
    end_date: str,
    out_dir: str
) -> Dict[str, str]:
    """
    Two-stage training:
      Stage-1 (forecast -> pseudo-observed nacelle signals)
      Stage-2 (predicted nacelle + forecast -> power)
    """
    set_seed(42)
    ensure_dir(out_dir)

    # 1) Load
    wx = load_weather_csvs(weather_files)
    scada = load_scada_excel(scada_path, start_date=start_date, end_date=end_date)

    # 2) Merge (WTG + Date/Time)
    df = merge_weather_scada(wx, scada)

    # 3) Basic transforms
    df = add_basic_time_features(df, time_col='Date/Time')
    df = add_diff_features(df, cols=['vws_10m','ta'], group_cols=['WTG'])
    df = wind_speed_binning(df, col='vws_10m')
    df = df.dropna().reset_index(drop=True)

    # 4) Split
    df_tr, df_te = time_order_split(df, ratio=0.8)

    # 5) Stage-1: Fit nacelle models using only forecast/time features
    stage1_feats = [c for c in ['vws_10m','uws_10m','ta','rh_1p5m','pmsl','dswrf','hour','day_of_week'] if c in df.columns]
    stage1_models = fit_stage1_models(df_tr, feature_cols=stage1_feats)

    # 6) Infer Stage-1 on both splits
    df_tr_s1 = infer_stage1(stage1_models, df_tr, stage1_feats)
    df_te_s1 = infer_stage1(stage1_models, df_te, stage1_feats)

    # 7) Stage-2: Build features (NO raw SCADA) & train
    X_tr = make_stage2_features(df_tr_s1)
    X_te = make_stage2_features(df_te_s1)
    y_tr = df_tr[TARGET_COL].values
    y_te = df_te[TARGET_COL].values

    power_model = fit_stage2_power(X_tr, y_tr)
    y_pred = power_model.predict(X_te)

    # 8) Metrics
    metrics = {
        'RMSE': rmse(y_te, y_pred),
        'MAE': mae(y_te, y_pred),
        'R2': r2(y_te, y_pred),
        'MAPE': safe_mape(y_te, y_pred),
        'NMAE': nmae(y_te, y_pred),
    }
    pd.DataFrame(metrics, index=[0]).to_csv(os.path.join(out_dir, 'metrics.csv'), index=False)

    # 9) Persist artifacts
    with open(os.path.join(out_dir, 'stage1_models.pkl'), 'wb') as f:
        pickle.dump(stage1_models, f)
    with open(os.path.join(out_dir, 'power_model.pkl'), 'wb') as f:
        pickle.dump(power_model, f)

    return {
        'stage1_models': os.path.join(out_dir, 'stage1_models.pkl'),
        'power_model': os.path.join(out_dir, 'power_model.pkl'),
        'metrics_csv': os.path.join(out_dir, 'metrics.csv'),
    }
