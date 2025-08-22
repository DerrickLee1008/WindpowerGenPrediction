
"""
Example inference script.
"""
import os
from utils.learning.test_part import predict_from_forecasts

ARTIFACT_DIR = "artifacts"

# Provide real paths here:
WEATHER_FILES = [
    # "Weather Data/WTG01_2306.csv",
]

if __name__ == "__main__":
    if not WEATHER_FILES:
        print("Please set WEATHER_FILES in scripts/test_example.py")
    else:
        df_pred = predict_from_forecasts(WEATHER_FILES, ARTIFACT_DIR)
        df_pred.to_csv("predictions.csv", index=False)
        print("Saved predictions.csv")
