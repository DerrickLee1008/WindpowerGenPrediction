
import os
import pickle
import pandas as pd
from typing import Dict, Tuple, List, Optional

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
    """(Legacy) Two-stage training runner without classes."""
    trainer = WindPowerTrainer(
        weather_files=weather_files,
        scada_path=scada_path,
        start_date=start_date,
        end_date=end_date,
        out_dir=out_dir
    )
    artifacts = trainer.run()
    return artifacts

class WindPowerTrainer:
    """
    Class-based orchestrator (close to windpowerprediction.py workflow):
      1) Load/merge
      2) Preprocess/feature engineering
      3) Stage-1 fit (forecast -> pseudo-observed SCADA)
      4) Stage-2 fit ((forecast + stage1_preds) -> power)
      5) Evaluate & persist artifacts
    """
    def __init__(
        self,
        weather_files: List[str],
        scada_path: str,
        start_date: str,
        end_date: str,
        out_dir: str,
        stage1_features: Optional[List[str]] = None,
        split_ratio: float = 0.8,
        seed: int = 42
    ) -> None:
        self.weather_files = weather_files
        self.scada_path = scada_path
        self.start_date = start_date
        self.end_date = end_date
        self.out_dir = out_dir
        self.split_ratio = split_ratio
        self.seed = seed

        self.stage1_features = stage1_features or [
            'vws_10m','uws_10m','ta','rh_1p5m','pmsl','dswrf','hour','day_of_week'
        ]

        self.df = None
        self.df_tr = None
        self.df_te = None
        self.stage1_models = None
        self.power_model = None
        self.metrics = None

    def run(self) -> Dict[str, str]:
        set_seed(self.seed)
        ensure_dir(self.out_dir)

        self._load_and_merge()
        self._preprocess()
        self._split()

        self._fit_stage1()
        self._create_stage1_preds()

        self._fit_stage2()
        self._evaluate()
        return self._persist()

    # steps
    def _load_and_merge(self) -> None:
        wx = load_weather_csvs(self.weather_files)
        scada = load_scada_excel(self.scada_path, start_date=self.start_date, end_date=self.end_date)
        self.df = merge_weather_scada(wx, scada)

    def _preprocess(self) -> None:
        self.df = add_basic_time_features(self.df, time_col='Date/Time')
        self.df = add_diff_features(self.df, cols=['vws_10m','ta'], group_cols=['WTG'])
        self.df = wind_speed_binning(self.df, col='vws_10m')
        self.df = self.df.dropna().reset_index(drop=True)

    def _split(self) -> None:
        self.df_tr, self.df_te = time_order_split(self.df, ratio=self.split_ratio)

    def _fit_stage1(self) -> None:
        feats = [c for c in self.stage1_features if c in self.df_tr.columns]
        self.stage1_models = fit_stage1_models(self.df_tr, feature_cols=feats)

    def _create_stage1_preds(self) -> None:
        feats_tr = [c for c in self.stage1_features if c in self.df_tr.columns]
        feats_te = [c for c in self.stage1_features if c in self.df_te.columns]
        self.df_tr = infer_stage1(self.stage1_models, self.df_tr, feats_tr)
        self.df_te = infer_stage1(self.stage1_models, self.df_te, feats_te)

    def _fit_stage2(self) -> None:
        X_tr = make_stage2_features(self.df_tr)
        X_te = make_stage2_features(self.df_te)
        y_tr = self.df_tr[TARGET_COL].values
        y_te = self.df_te[TARGET_COL].values

        # Leakage guard
        forbidden = {
            'Nacelle\nWind Speed\n[m/s]',
            'Nacelle\nWind Direction\n[deg]',
            'Nacelle\nOutdoor Temp\n[℃]',
            'Rotor\nRotor Speed\n[rpm]'
        }
        leaks = forbidden.intersection(set(X_tr.columns))
        assert not leaks, f"SCADA leakage detected in Stage-2 features: {leaks}"

        self.power_model = fit_stage2_power(X_tr, y_tr)
        self._X_te = X_te
        self._y_te = y_te

    def _evaluate(self) -> None:
        y_pred = self.power_model.predict(self._X_te)
        self.metrics = {
            'RMSE': rmse(self._y_te, y_pred),
            'MAE': mae(self._y_te, y_pred),
            'R2': r2(self._y_te, y_pred),
            'MAPE': safe_mape(self._y_te, y_pred),
            'NMAE': nmae(self._y_te, y_pred),
        }

    def _persist(self) -> Dict[str, str]:
        pd.DataFrame(self.metrics, index=[0]).to_csv(os.path.join(self.out_dir, 'metrics.csv'), index=False)
        with open(os.path.join(self.out_dir, 'stage1_models.pkl'), 'wb') as f:
            pickle.dump(self.stage1_models, f)
        with open(os.path.join(self.out_dir, 'power_model.pkl'), 'wb') as f:
            pickle.dump(self.power_model, f)
        return {
            'stage1_models': os.path.join(self.out_dir, 'stage1_models.pkl'),
            'power_model': os.path.join(self.out_dir, 'power_model.pkl'),
            'metrics_csv': os.path.join(self.out_dir, 'metrics.csv'),
        }
