"""
Project Configuration File

All global configuration such as directory paths, Hyperparameters etc should be defined here.
"""

from pathlib import Path

# BASE Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# print(f"BASE Directory: {BASE_DIR}")

# Data Directory
DATA_DIR = BASE_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

RAW_DATA_PATH = RAW_DATA_DIR / "fake_job_postings.csv"

PROCESSED_DATA_DIR = DATA_DIR / "Preprocessed"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "preprocessed_data.csv"

# Train Hyperparameter

RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET_COLUMN = "fraudulent"
