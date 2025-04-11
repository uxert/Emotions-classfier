"""
Top-level package for Emotions Classifier. This contains the PyTorch model class, everything needed to download and
set up the dataset and download an already trained EmotionRecognizer model. Does NOT contain functionality to train
the model.
"""

__all__ = ["EmotionRecognizer", "RAVDESSDataset", "RAVDESS_DATASET_DIR", "RAVDESS_ZIP_PATH", "utils"]

import os.path

PACKAGE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_DIR)

RAVDESS_ZIP_PATH = os.path.join(PROJECT_ROOT, "data", "ravdess.zip")
RAVDESS_DATASET_DIR = os.path.join(PROJECT_ROOT, "ravdess_data")

from .EmotionRecognizer import EmotionRecognizer
from .RAVDESSDataset import RAVDESSDataset
