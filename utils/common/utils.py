
import os
import random
import numpy as np
import pandas as pd

def set_seed(seed: int = 42):
    """
    Set seeds for numpy / python / torch / tf to improve reproducibility.
    """
    import importlib
    random.seed(seed)
    np.random.seed(seed)
    # torch
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    # tensorflow
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def time_features(dt_series: pd.Series) -> pd.DataFrame:
    s = pd.to_datetime(dt_series)
    return pd.DataFrame({
        "hour": s.dt.hour,
        "day": s.dt.day,
        "month": s.dt.month,
        "day_of_week": s.dt.dayofweek,
    }, index=s.index)

def safe_float_convert(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
