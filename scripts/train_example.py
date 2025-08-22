
"""
Example training script.
Adjust `WEATHER_FILES` and `SCADA_PATH` to your local paths.
"""
import os
from utils.learning.train_part import train_pipeline

OUT_DIR = "artifacts"

# Provide real paths here:
WEATHER_FILES = [
    # "Weather Data/WTG01_2303.csv",
    # "Weather Data/WTG01_2304.csv",
    # "Weather Data/WTG01_2305.csv",
]

SCADA_PATH = "scada_gyeongju_2023_10min.xlsx"

if __name__ == "__main__":
    if not WEATHER_FILES:
        print("Please set WEATHER_FILES in scripts/train_example.py")
    else:
        os.makedirs(OUT_DIR, exist_ok=True)
        artifacts = train_pipeline(
            WEATHER_FILES,
            SCADA_PATH,
            start_date="2023-03-01",
            end_date="2023-05-31",
            out_dir=OUT_DIR
        )
        print("Artifacts:", artifacts)
