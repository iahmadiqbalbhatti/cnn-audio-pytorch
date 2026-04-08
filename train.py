from pathlib import Path

import modal
import numpy as np
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchaudio.transforms as T
from torch.xpu import device
from tqdm import tqdm

from model import AudioCNN

app = modal.App("audio-cnn")

image = (modal.Image.debian_slim()
         .pip_install_from_requirements("requirements.txt")
         .apt_install(["wget", "unzip", "ffmpeg", "libsndfile1"])
         .run_commands([
    "cd /tmp && wget https://github.com/karolpiczak/ESC-50/archive/master.zip -O esc50.zip",
    "cd /tmp && unzip esc50.zip",
    "mkdir -p /opt/esc50-data",
    "cp -r /tmp/ESC-50-master/* /opt/esc50-data/",
    "rm -rf /tmp/esc50.zip /tmp/ESC-50-master"
])
         .add_local_python_source("model"))

volume = modal.Volume.from_name("esc50-data", create_if_missing=True)

model_volume = modal.Volume.from_name("esc-model", create_if_missing=True)


class ESC50Dataset(Dataset):
    def __init__(self, data_dir, metadata_file, split="train", transform=None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.metadata_file = pd.read_csv(metadata_file)
        self.split = split
        self.transform = transform

        if split == "train":
            self.metadata_file = self.metadata_file[self.metadata_file['fold'] != 5]
        else:
            self.metadata_file = self.metadata_file[self.metadata_file['fold'] == 5]

        self.classes = sorted(self.metadata_file['category'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.metadata_file['label'] = self.metadata_file['category'].map(lambda x: self.class_to_idx[x])

    def __len__(self):
        return len(self.metadata_file)

    def __getitem__(self, idx):
        row = self.metadata_file.iloc[idx]
        audio_path = self.data_dir / "audio" / row['firename']  # data_dir/audio/filename.wav

        waveform, sample_rate = torchaudio.load(audio_path)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if self.transform:
            spectrogram = self.transform(waveform)

        else:
            spectrogram = waveform

        return spectrogram, row['label']

    @staticmethod
    def mix_up_data(x, y):
        lam = np.random.beta(0.4, 0.4)
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam

    @staticmethod
    def mix_up_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


@app.function(
    image=image,
    gpu="T4",
    volumes={"/data": volume, "/models": model_volume},
    timeout=3600,
)
def train():
    print("Training model...")
    print(torch.__version__)
    esc50_dir = Path("/opt/esc50-data")

    train_transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            f_min=0,
            f_max=11025
        ),
        T.AmplitudeToDB(),
        T.FrequencyMasking(freq_mask_param=30),
        T.TimeMasking(time_mask_param=80),
    )

    val_transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            f_min=0,
            f_max=11025
        ),
        T.AmplitudeToDB()
    )

    train_dataset = ESC50Dataset(data_dir=esc50_dir, metadata_file=esc50_dir / 'meta' / "esc50.csv", split="train",
                                 transform=train_transform)
    val_dataset = ESC50Dataset(data_dir=esc50_dir, metadata_file=esc50_dir / 'meta' / "esc50.csv", split="val",
                               transform=val_transform)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioCNN(num_classes=len(train_dataset.classes)).to(device)

    num_epochs = 100
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.002,
        steps_per_epoch=len(train_dataloader),
        epochs=num_epochs,
        pct_start=0.1,
    )

    best_accuracy = 0.0

    print("Starting training...")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)

            if np.random.rand() > 0.7:
                data, target_a, target_b, lam = ESC50Dataset.mix_up_data(data, target)
                output = model(data)
                loss = ESC50Dataset.mix_up_criterion(criterion, output, target_a, target_b, lam)
            else:
                output = model(data)
                loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        # Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_dataloader):.4f}, Accuracy: {accuracy:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "/models/best_model.pt")
            print(f"Model saved with accuracy: {best_accuracy:.2f}%")


@app.local_entrypoint()
def main():
    train.remote()