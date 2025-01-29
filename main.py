import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from dataset_utils import unzip_nested_dataset, load_ravdess_data
from EmotionRecognizer import EmotionRecognizer
from RAVDESSDataset import RAVDESSDataset

MODEL_PATH = "EmotionsClassifier.pth"
MODEL_URL = "https://drive.google.com/uc?id=1rEBqU3geg2V4_W9QGY-nZGWWdNXd4zrq"


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

if __name__ == "__main__":
    main()
    