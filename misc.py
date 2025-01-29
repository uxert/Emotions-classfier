from playsound import playsound
import torch
from RAVDESSDataset import RAVDESSDataset
from EmotionRecognizer import EmotionRecognizer
import gdown
import os


def display_audio_and_predictions(model, dataset: RAVDESSDataset, class_names, num_samples=5):
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Displaying audio files and model predictions:")
    for i in range(num_samples):
        # Select a random sample from the dataset
        idx = torch.randint(0, len(dataset), (1,)).item()
        mel_spec, label = dataset[idx]

        # Pass the sample through the model
        mel_spec = mel_spec.unsqueeze(0).to(device)  # Add batch dimension
        with torch.no_grad():
            output = model(mel_spec)
            predicted_class = output.argmax(dim=1).item()

        # Display the prediction
        print(f"\nSample {i + 1}/{num_samples}")
        print(f"True Emotion: {class_names[label]}")
        print(f"Predicted Emotion: {class_names[predicted_class]}")

        # Play the audio file
        audio_path = dataset.file_paths[idx]
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




