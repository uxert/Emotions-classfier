from typing import Tuple
import gdown
import os
import zipfile
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from emotions_classifier import RAVDESSDataset


def download_dataset_from_gdrive(google_drive_url: str, dataset_path: str):
    if os.path.exists(dataset_path):
        print("Already downloaded!")
        return
    print("Downloading...")
    dataset_dir = os.path.dirname(dataset_path)
    os.makedirs(dataset_dir, exist_ok=True)
    gdown.download(google_drive_url, dataset_path, quiet=False)

def unzip_nested_dataset(main_zip_path, extract_to):
    # Extract main archive
    with zipfile.ZipFile(main_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    # Locate and extract nested archives (e.g., speech and song zips)
    nested_zips = [f for f in os.listdir(extract_to) if f.endswith('.zip')]
    for nested_zip in nested_zips:
        nested_zip_path = os.path.join(extract_to, nested_zip)
        with zipfile.ZipFile(nested_zip_path, 'r') as zip_ref:
            nested_extract_to = os.path.join(extract_to, os.path.splitext(nested_zip)[0])
            zip_ref.extractall(nested_extract_to)
        print(f"Extracted {nested_zip} to {nested_extract_to}")


def load_ravdess_data(data_dir, audio_type="speech"):
    """
    Load RAVDESS data from the extracted directories.
    Args:
        data_dir (str): Root directory of the extracted RAVDESS dataset.
        audio_type (str): "speech" or "song" to choose the corresponding audio data.
    Returns:
        file_paths (list): List of audio file paths.
        labels (list): Corresponding emotion labels for the audio files.
    """
    subfolder = "Audio_Speech_Actors_01-24" if audio_type == "speech" else "Audio_Song_Actors_01-24"
    full_path = os.path.join(data_dir, subfolder)

    file_paths = []
    labels = []

    # Walk through the directory to find audio files
    for root, _, files in os.walk(full_path):
        for file in files:
            if file.endswith(".wav"):
                file_paths.append(os.path.join(root, file))
                # Extract emotion label (e.g., "03" for happy)
                labels.append(int(file.split("-")[2]) - 1)  # Zero-indexed

    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    return file_paths, labels

def split_train_test_val(file_paths, labels, seed=42) -> Tuple[RAVDESSDataset, RAVDESSDataset, RAVDESSDataset]:
    """
    Splits into 60% train, 20% test and 20% val, always with given random state - ensuring reproductibility.
    Published, already trained model was trained on a dataset divided with seed 42.

    """
    train_files, test_files, train_labels, test_labels = train_test_split(
        file_paths, labels, test_size=0.2, stratify=labels, random_state=seed
    )
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_files, train_labels, test_size=0.25, stratify=train_labels, random_state=seed
    )

    # Create datasets and dataloaders
    train_dataset = RAVDESSDataset(train_files, train_labels, train_mode=True)
    val_dataset = RAVDESSDataset(val_files, val_labels)
    test_dataset = RAVDESSDataset(test_files, test_labels)

    return train_dataset, test_dataset, val_dataset