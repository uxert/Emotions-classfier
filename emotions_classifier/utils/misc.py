from playsound import playsound
import torch
from emotions_classifier import EmotionRecognizer, RAVDESSDataset
import gdown
import os


def display_audio_and_predictions(model: EmotionRecognizer, dataset: RAVDESSDataset, class_names, num_samples=5):
    """
    Displays audio files and prints model predictions (console version with audio playback).

    Args:
        model: Trained PyTorch model.
        dataset: Dataset object (e.g., RAVDESSDataset) subclassing torch.utils.data.Dataset.
        class_names: List of emotion class names corresponding to the labels.
        num_samples: Number of audio samples to display and predict.
    """
    # Set the model to evaluation mode
    model.eval()

    print("Displaying audio files and model predictions:")
    for i in range(num_samples):
        # Select a random sample from the dataset
        true_emotion, predicted_emotion, audio_path = model.make_one_prediction(dataset, class_names)
        # Display the prediction
        print(f"\nSample {i + 1}/{num_samples}")
        print(f"True Emotion: {true_emotion}")
        print(f"Predicted Emotion: {predicted_emotion}")

        # Play the audio file
        print(f"Playing audio file: {audio_path}")
        playsound(audio_path)

def download_model_from_gdrive(model_url, model_path):
    gdown.download(model_url, model_path, quiet=False)

def load_model_for_inference(model_url, model_path, device="cpu") -> EmotionRecognizer:
    if not os.path.exists(model_path):
        download_model_from_gdrive(model_url, model_path)
    else:
        print("Model already downloaded :)")
    model = EmotionRecognizer()
    _ = model(EmotionRecognizer.dummy_input)  # initializes lazy layers before loading
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.to(device)
    model.eval()
    return model




