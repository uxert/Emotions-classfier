from emotions_classifier.utils import unzip_nested_dataset, load_ravdess_data, RAVDESS_DOWNLOAD_URL, \
    download_dataset_from_gdrive, load_model_for_inference, split_train_test_val
from emotions_classifier import RAVDESS_ZIP_PATH, RAVDESS_DATASET_DIR


MODEL_PATH = "EmotionsClassifier.pth"
MODEL_URL = "https://drive.google.com/uc?id=1rEBqU3geg2V4_W9QGY-nZGWWdNXd4zrq"


def inference():
    download_dataset_from_gdrive(RAVDESS_DOWNLOAD_URL, RAVDESS_ZIP_PATH)
    class_names = [
        "Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"
    ]
    unzip_nested_dataset(RAVDESS_ZIP_PATH, RAVDESS_DATASET_DIR)
    file_paths, labels = load_ravdess_data(RAVDESS_DATASET_DIR, audio_type="speech")

    _, test_dataset, _ = split_train_test_val(file_paths, labels)

    my_model = load_model_for_inference(MODEL_URL, MODEL_PATH)
    from GUI import MyGUI
    gui = MyGUI()
    gui.show_gui(my_model, test_dataset, class_names)

if __name__ == "__main__":
    inference()
