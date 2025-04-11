import os.path

PACKAGE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_DIR)

RAVDESS_ZIP_PATH = os.path.join(PROJECT_ROOT, "data", "ravdess.zip")
RAVDESS_DATASET_DIR = os.path.join(PROJECT_ROOT, "ravdess_data")