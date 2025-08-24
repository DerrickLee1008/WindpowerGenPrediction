
"""Example inference script (class-based)."""
import os
from utils.learning.test_part import WindPowerPredictor

ARTIFACT_DIR = "artifacts"
WEATHER_FILES = [
    # "Weather Data/WTG01_2306.csv",
]

if __name__ == "__main__":
    if not WEATHER_FILES:
        print("Please set WEATHER_FILES in scripts/test_example.py")
    else:
        predictor = WindPowerPredictor(artifact_dir=ARTIFACT_DIR)
        predictor.load()
        df_pred = predictor.predict(WEATHER_FILES)
        df_pred.to_csv("predictions.csv", index=False)
        print("Saved predictions.csv")
