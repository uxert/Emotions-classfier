from torch import zeros, inference_mode, randint
import torch.nn as nn
import torch.nn.functional as F
from emotions_classifier.RAVDESSDataset import RAVDESSDataset

class EmotionRecognizer(nn.Module):

    dummy_input = zeros((32,1,64,94), device='cpu')
    def __init__(self, dropout_prob = 0.4, conv_dropout_prob = 0.3):
        super(EmotionRecognizer, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn_conv1 = nn.LazyBatchNorm2d()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn_conv2 = nn.LazyBatchNorm2d()
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn_conv3 = nn.LazyBatchNorm2d()

        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.LazyLinear(out_features=256)
        self.bn_fc_1 = nn.LazyBatchNorm1d()
        self.fc2 = nn.LazyLinear(out_features=8)  # 8 emotion classes

        self.dropout = nn.Dropout(p=dropout_prob)
        self.conv_dropout = nn.Dropout2d(p=conv_dropout_prob/3)
        self.after_conv_dropout = nn.Dropout2d(p=conv_dropout_prob)

    def forward(self, x):
        x = self.pool(F.relu(self.bn_conv1(self.conv1(x))))
        x = self.conv_dropout(x)
        x = self.pool(F.relu(self.bn_conv2(self.conv2(x))))
        x = self.conv_dropout(x)
        x = self.pool(F.relu(self.bn_conv3(self.conv3(x))))
        x = self.after_conv_dropout(x)
        x = x.view(x.size(0), -1)  # Flatten

        x = F.relu(self.fc1(x))
        x = self.bn_fc_1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # no softmax here - it happens in CrossEntropyLoss
        return x

    def make_one_prediction(self, dataset: RAVDESSDataset, class_names) -> tuple[str, str, str]:
        """
        :return: true emotion, predicted emotion, audio path (to play the sound with playsound)
        """
        with inference_mode():
            idx = randint(0, len(dataset), (1,)).item()
            mel_spec, label = dataset[idx]

            mel_spec = mel_spec.unsqueeze(0)
            output = self(mel_spec)
            predicted_class = output.argmax(dim=1).item()

            # Display the prediction
            true_emotion = class_names[label]
            predicted_emotion = class_names[predicted_class]

            audio_path = dataset.file_paths[idx]
            return true_emotion, predicted_emotion, audio_path


