import gdown
import os


RAVDESS_DOWNLOAD_URL = "https://drive.google.com/uc?id=1NlnM16W96WMspro-uLqvul0dlNIGSajF"
RAVDESS_PATH = "data/ravdess.zip"

def download_dataset_from_gdrive(google_drive_url: str, dataset_path: str):
    if os.path.exists(dataset_path):
        if len(os.listdir(dataset_path)) != 0:
            return  # we do nothing if the dataset is already downloaded
    print("Downloading...")
    gdown.download(google_drive_url, dataset_path, quiet=False)

download_dataset_from_gdrive(RAVDESS_DOWNLOAD_URL, RAVDESS_PATH)