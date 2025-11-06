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

TRAIN_FEATURES_PATH = PROCESSED_DATA_DIR / "train_features.npz"
TEST_FEATURES_PATH = PROCESSED_DATA_DIR / "test_features.npz"

TRAIN_TARGET_PATH = PROCESSED_DATA_DIR / "train_target.npy"
TEST_TARGET_PATH = PROCESSED_DATA_DIR / "test_target.npy"

# Train Hyperparameter

RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET_COLUMN = "fraudulent"
