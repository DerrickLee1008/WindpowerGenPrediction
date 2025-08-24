
"""Example training script (class-based)."""
import os
from utils.learning.train_part import WindPowerTrainer

OUT_DIR = "artifacts"

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
        trainer = WindPowerTrainer(
            weather_files=WEATHER_FILES,
            scada_path=SCADA_PATH,
            start_date="2023-03-01",
            end_date="2023-05-31",
            out_dir=OUT_DIR
        )
        artifacts = trainer.run()
        print("Artifacts:", artifacts)
