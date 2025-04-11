import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from emotions_classifier import EmotionRecognizer, RAVDESSDataset, RAVDESS_ZIP_PATH, RAVDESS_DATASET_DIR
from emotions_classifier.utils import unzip_nested_dataset, load_ravdess_data, download_dataset_from_gdrive
from sklearn.model_selection import train_test_split

from main import MODEL_PATH, RAVDESS_DOWNLOAD_URL

EPOCHS = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-5  # for Adam optimizer

def train_and_save_model(model, train_loader: DataLoader, val_loader, epochs=10, lr=0.001, weight_decay=1e-5):
    """Trains and saves the model, not much more to say here"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


def test_model(model, test_loader: DataLoader):
    """Tests the model on provided DataLoader"""
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

def train_save_test_model():
    # Download the dataset, if not already downloaded
    download_dataset_from_gdrive(RAVDESS_DOWNLOAD_URL, RAVDESS_ZIP_PATH)
    # Extract the dataset
    unzip_nested_dataset(RAVDESS_ZIP_PATH, RAVDESS_DATASET_DIR)

    # Load speech data (can also switch to "song")
    file_paths, labels = load_ravdess_data(RAVDESS_DATASET_DIR, audio_type="speech")

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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Instantiate model
    model = EmotionRecognizer()

    # Train the model
    train_and_save_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Test the model
    my_model = test_model(model, test_loader)

if __name__ == "__main__":
    train_save_test_model()