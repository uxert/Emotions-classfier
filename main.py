import os
import zipfile
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from misc import display_audio_and_predictions
from EmotionRecognizer import EmotionRecognizer
from RAVDESSDataset import RAVDESSDataset


# --- Step 1: Unzip the Dataset ---
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


# --- Step 2: Load Data ---
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

def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for mel_spec, labels in train_loader:
            mel_spec, labels = mel_spec.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(mel_spec)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for mel_spec, labels in val_loader:
                mel_spec, labels = mel_spec.to(device), labels.to(device)
                outputs = model(mel_spec)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        print(
            f"Epoch {epoch + 1}: Train Loss = {train_loss / len(train_loader):.4f}, Val Loss = {val_loss / len(val_loader):.4f}")


def test_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for mel_spec, labels in test_loader:
            mel_spec, labels = mel_spec.to(device), labels.to(device)
            outputs = model(mel_spec)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)

            # noinspection PyUnresolvedReferences
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {correct / total * 100:.2f}%, for total of {total} predictions")
    return model

def main():
    # Extract the dataset
    unzip_nested_dataset("data/ravdess.zip", "ravdess_data")

    # Load speech data (you can switch to "song" if needed)
    file_paths, labels = load_ravdess_data("ravdess_data", audio_type="speech")

    # --- Step 3: Split Data ---
    train_files, test_files, train_labels, test_labels = train_test_split(
        file_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_files, train_labels, test_size=0.25, stratify=train_labels, random_state=42
    )

    # Create datasets and dataloaders
    train_dataset = RAVDESSDataset(train_files, train_labels, train_mode=True)
    val_dataset = RAVDESSDataset(val_files, val_labels)
    test_dataset = RAVDESSDataset(test_files, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Instantiate model
    model = EmotionRecognizer()

    # Train the model
    train_model(model, train_loader, val_loader)

    # Test the model
    my_model = test_model(model, test_loader)

    class_names = [
        "Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"
    ]

    # display_audio_and_predictions(my_model, test_dataset, class_names, num_samples=5)

if __name__ == "__main__":
    main()
    