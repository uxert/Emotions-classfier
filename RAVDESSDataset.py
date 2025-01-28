from torch.utils.data import Dataset
import torchaudio
import torch
import torch.nn.functional as F

class RAVDESSDataset(Dataset):
    def __init__(self, file_paths, labels, sample_rate=16000, max_duration=3):
        self.file_paths = file_paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.max_length = sample_rate * max_duration  # Maximum length in samples

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load audio file
        waveform, sample_rate = torchaudio.load(self.file_paths[idx])

        # Resample if necessary
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Truncate or pad
        if waveform.shape[1] > self.max_length:
            waveform = waveform[:, :self.max_length]
        else:
            waveform = F.pad(waveform, (0, self.max_length - waveform.shape[1]))

        # Compute Mel Spectrogram
        mel_spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=64,
            n_fft=1024,
            hop_length=512
        )
        mel_spec = mel_spec_transform(waveform)

        # Normalize and take log scale
        mel_spec = torch.log(mel_spec + 1e-9)
        mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()

        # Return features and label
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return mel_spec, label
