from sklearn.model_selection import train_test_split
from emotions_classifier.utils import unzip_nested_dataset, load_ravdess_data, RAVDESS_DOWNLOAD_URL, \
    download_dataset_from_gdrive, load_model_for_inference
from emotions_classifier import RAVDESS_ZIP_PATH, RAVDESS_DATASET_DIR
from emotions_classifier.RAVDESSDataset import RAVDESSDataset

MODEL_PATH = "EmotionsClassifier.pth"
MODEL_URL = "https://drive.google.com/uc?id=1rEBqU3geg2V4_W9QGY-nZGWWdNXd4zrq"


def inference():
    download_dataset_from_gdrive(RAVDESS_DOWNLOAD_URL, RAVDESS_ZIP_PATH)
    class_names = [
        "Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"
    ]
    unzip_nested_dataset(RAVDESS_ZIP_PATH, RAVDESS_DATASET_DIR)
    file_paths, labels = load_ravdess_data(RAVDESS_DATASET_DIR, audio_type="speech")

    # Split the dataset in the exact same way as with training
    train_files, _, train_labels, _ = train_test_split(
        file_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    _, val_files, _, val_labels = train_test_split(
        train_files, train_labels, test_size=0.25, stratify=train_labels, random_state=42
    )

    dataset = RAVDESSDataset(val_files, val_labels)

    my_model = load_model_for_inference(MODEL_URL, MODEL_PATH)
    from GUI import MyGUI
    gui = MyGUI()
    gui.show_gui(my_model, dataset, class_names)

if __name__ == "__main__":
    inference()
