import torch
torch.backends.cudnn.benchmark = False
from torch.amp import autocast, GradScaler
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from decord import VideoReader, cpu
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os
import re
import glob
from functools import partial

# =====================
# CONFIGURATION
# =====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UCF_ROOT = os.path.join(BASE_DIR, "data", "UCF101")
ANNOTATION_DIR = os.path.join(BASE_DIR, "data", "UCF101TrainTestSplits-RecognitionTask")
SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videomae_ucf101.pth")
CHECKPOINT_PREFIX = "videomae_ucf101_epoch"
NUM_EPOCHS = 5
NUM_FRAMES = 16
BATCH_SIZE = 2                # VideoMAE-base is VRAM-heavy, kept small (consistent with the demo run)
GRAD_ACCUM_STEPS = 4          # effective batch size = 2*4 = 8
NUM_WORKERS = 4
# =====================


class UCF101Dataset(Dataset):
    def __init__(self, annotation_file, ucf_root, num_frames=16):
        self.ucf_root = ucf_root
        self.num_frames = num_frames
        self.label2id = {}
        self.samples = []

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
                return None, label
            # Full-video uniform sampling, consistent with R(2+1)D / I3D scripts
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            frames = vr.get_batch(indices).asnumpy()
        except Exception:
            return None, label
        return list(frames), label  # raw frame list, processor handles resize/normalize


def collate_fn(batch, processor):
    frames_list, labels = zip(*batch)
    valid = [(f, l) for f, l in zip(frames_list, labels) if f is not None]
    if not valid:
        return None, None
    frames_list, labels = zip(*valid)
    inputs = processor(list(frames_list), return_tensors="pt")
    return inputs["pixel_values"], torch.tensor(labels)


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading MCG-NJU/videomae-base (Kinetics-400 pretrained)...")
    processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    model = VideoMAEForVideoClassification.from_pretrained(
        "MCG-NJU/videomae-base",
        num_labels=101,
        ignore_mismatched_sizes=True  # classification head is randomly initialized for 101 UCF101 classes
    )
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
        collate_fn=partial(collate_fn, processor=processor),
        pin_memory=True if torch.cuda.is_available() else False
    )

    optimizer = AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler('cuda')

    CHECKPOINT_DIR = os.path.dirname(SAVE_PATH)

    # --- Automatic checkpoint detection / resume ---
    start_epoch = 0
    latest_checkpoint = find_latest_checkpoint(CHECKPOINT_DIR)
    if latest_checkpoint is not None:
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Auto-resume: found {latest_checkpoint}, starting at epoch {start_epoch + 1}")
    else:
        print("No checkpoint found, starting from scratch.")
    # --- /Automatic checkpoint detection ---

    if start_epoch >= NUM_EPOCHS:
        print(f"Training already completed (epoch {start_epoch}/{NUM_EPOCHS}). Nothing to do.")
        return

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        skipped_batches = 0

        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for batch_idx, (pixel_values, labels) in enumerate(pbar):
            try:
                if pixel_values is None:
                    skipped_batches += 1
                    continue

                pixel_values = pixel_values.to(device)
                labels = labels.to(device)

                with autocast('cuda'):
                    outputs = model(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss / GRAD_ACCUM_STEPS

                scaler.scale(loss).backward()

                if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                total_loss += outputs.loss.item()
                predicted = outputs.logits.argmax(-1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            except RuntimeError as e:
                print(f"[WARNING] Batch {batch_idx} skipped — {e}")
                skipped_batches += 1
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                continue

            if batch_idx % 50 == 0:
                pbar.set_postfix({'loss': f"{total_loss/max(batch_idx+1,1):.4f}"})

        avg_loss = total_loss / max(len(train_loader) - skipped_batches, 1)
        print(f"Epoch {epoch+1} — Avg Loss: {avg_loss:.4f} — "
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