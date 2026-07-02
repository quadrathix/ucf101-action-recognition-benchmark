import torch
torch.backends.cudnn.benchmark = False
import torchvision
import numpy as np
import os
import re
import glob
from decord import VideoReader, cpu
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from torchvision import transforms

# =====================
# CONFIGURATION
# =====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UCF_ROOT = os.path.join(BASE_DIR, "data", "UCF101")
ANNOTATION_DIR = os.path.join(BASE_DIR, "data", "UCF101TrainTestSplits-RecognitionTask")
SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "i3d_ucf101.pth")
NUM_EPOCHS = 5
BATCH_SIZE = 8
NUM_WORKERS = 4
NUM_FRAMES = 8          # i3d_r50'nin resmi konfigürasyonu: 8 frame, 8x8 sampling
CROP_SIZE = 224          # Resmi tutorial 256x256 kullanıyor; VRAM kısıtı nedeniyle 224x224'e düşürüldü
CHECKPOINT_PREFIX = "i3d_ucf101_epoch"
# =====================

class UCF101Dataset(Dataset):
    def __init__(self, annotation_file, ucf_root, num_frames=8):
        self.ucf_root = ucf_root
        self.num_frames = num_frames
        self.label2id = {}
        self.samples = []
        self.resize_transform = transforms.Resize((CROP_SIZE, CROP_SIZE))

        with open(annotation_file, "r") as f:
            for line in f:
                parts = line.strip().split(" ")
                if not parts or parts[0] == "":
                    continue
                video_rel_path = parts[0]
                label_name = video_rel_path.split("/")[0]

                if label_name not in self.label2id:
                    self.label2id[label_name] = len(self.label2id)

                self.samples.append((video_rel_path, self.label2id[label_name]))

        self.id2label = {v: k for k, v in self.label2id.items()}
        print(f"Dataset loaded — Videos: {len(self.samples)}, Classes: {len(self.label2id)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_rel_path, label = self.samples[idx]
        video_path = os.path.join(self.ucf_root, video_rel_path)

        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)

            if total_frames < self.num_frames:
                return torch.zeros((3, self.num_frames, CROP_SIZE, CROP_SIZE)), label

            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            frames = vr.get_batch(indices).asnumpy()

        except Exception:
            return torch.zeros((3, self.num_frames, CROP_SIZE, CROP_SIZE)), label

        video = torch.tensor(frames).permute(3, 0, 1, 2).float() / 255.0
        video = self.resize_transform(video)

        mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
        std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)
        video = (video - mean) / std

        return video, label


def find_latest_checkpoint(checkpoint_dir, prefix=CHECKPOINT_PREFIX):
    pattern = os.path.join(checkpoint_dir, f"{prefix}*.pth")
    candidates = glob.glob(pattern)
    if not candidates:
        return None

    def extract_epoch(path):
        match = re.search(rf"{prefix}(\d+)\.pth$", os.path.basename(path))
        return int(match.group(1)) if match else -1

    latest = max(candidates, key=extract_epoch)
    return latest if extract_epoch(latest) > 0 else None


def main():
    print("Loading pretrained I3D (i3d_r50) model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.hub.load("facebookresearch/pytorchvideo", "i3d_r50", pretrained=True)
    model.blocks[6].proj = torch.nn.Linear(2048, 101)
    model = model.to(device)

    train_dataset = UCF101Dataset(
        annotation_file=os.path.join(ANNOTATION_DIR, "trainlist01.txt"),
        ucf_root=UCF_ROOT,
        num_frames=NUM_FRAMES
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )

    optimizer = AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    CHECKPOINT_DIR = os.path.dirname(SAVE_PATH)

    # --- Otomatik checkpoint algılama ---
    start_epoch = 0
    latest_checkpoint = find_latest_checkpoint(CHECKPOINT_DIR)
    if latest_checkpoint is not None:
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Auto continue: {latest_checkpoint} found, starting from epoch {start_epoch + 1}")
    else:
        print("Checkpoint not found, initializing")
    # --- /Otomatik checkpoint algılama ---

    # Zaten tamamlanmışsa gereksiz yere tekrar çalışmayı/final'i tekrar kaydetmeyi engelle
    if start_epoch >= NUM_EPOCHS:
        print(f"Training already complete (epoch {start_epoch}/{NUM_EPOCHS}).")
        return

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        skipped_batches = 0

        for batch_idx, (videos, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")):
            try:
                videos = videos.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(videos)

                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    skipped_batches += 1
                    continue

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                predicted = outputs.argmax(-1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            except RuntimeError as e:
                print(f"[UYARI] Batch {batch_idx} atlandı — {e}")
                skipped_batches += 1
                torch.cuda.empty_cache()
                continue

        print(f"Epoch {epoch+1} — Avg Loss: {total_loss/len(train_loader):.4f} — "
              f"Train Accuracy: {correct/max(total,1)*100:.2f}% — Skipped batches: {skipped_batches}")

        epoch_save_path = os.path.join(CHECKPOINT_DIR, f"{CHECKPOINT_PREFIX}{epoch+1}.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'label2id': train_dataset.label2id,
            'id2label': train_dataset.id2label,
            'epoch': epoch + 1
        }, epoch_save_path)
        print(f"Checkpoint saved: {epoch_save_path}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'label2id': train_dataset.label2id,
        'id2label': train_dataset.id2label
    }, SAVE_PATH)
    print(f"Model successfully saved to {SAVE_PATH}")


if __name__ == '__main__':
    main()