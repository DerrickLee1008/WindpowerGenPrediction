
import numpy as np

def rmse(y_true, y_pred):
    y = np.asarray(y_true); p = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y - p) ** 2)))

def mae(y_true, y_pred):
    y = np.asarray(y_true); p = np.asarray(y_pred)
    return float(np.mean(np.abs(y - p)))

def r2(y_true, y_pred):
    y = np.asarray(y_true); p = np.asarray(y_pred)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-12
    return float(1 - ss_res / ss_tot)

def safe_mape(y_true, y_pred, eps=1e-6):
    y = np.asarray(y_true); p = np.asarray(y_pred)
    denom = np.maximum(np.abs(y), eps)
    return float(np.mean(np.abs((y - p) / denom)) * 100.0)

def nmae(y_true, y_pred):
    y = np.asarray(y_true); p = np.asarray(y_pred)
    return float(np.mean(np.abs(y - p)) / (np.mean(np.abs(y)) + 1e-6))
