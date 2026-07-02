import torch
from torch.cuda.amp import autocast, GradScaler
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from decord import VideoReader, cpu
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os
import time

from functools import partial

# =====================
# CONFIGURATION
# =====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UCF_ROOT = os.path.join(BASE_DIR, "data", "UCF101")
ANNOTATION_DIR = os.path.join(BASE_DIR, "data", "UCF101TrainTestSplits-RecognitionTask")
NUM_FRAMES = 16
BATCH_SIZE = 2                # VideoMAE-base is VRAM-heavy, starting small
GRAD_ACCUM_STEPS = 4          # effective batch size = 2*4 = 8
NUM_DEMO_BATCHES = 200        # sample size used to estimate full-epoch duration without running it
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
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            frames = vr.get_batch(indices).asnumpy()
        except Exception:
            return None, label
        return list(frames), label  # pass raw frame list to the processor
    
def collate_fn(batch, processor):
    frames_list, labels = zip(*batch)
    valid = [(f, l) for f, l in zip(frames_list, labels) if f is not None]
    frames_list, labels = zip(*valid)
    inputs = processor(list(frames_list), return_tensors="pt")
    return inputs["pixel_values"], torch.tensor(labels)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading MCG-NJU/videomae-base (Kinetics-400 pretrained)...")
    processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    model = VideoMAEForVideoClassification.from_pretrained(
        "MCG-NJU/videomae-base",
        num_labels=101,
        ignore_mismatched_sizes=True
    )
    model = model.to(device)
    model.train()

    train_dataset = UCF101Dataset(
        annotation_file=os.path.join(ANNOTATION_DIR, "trainlist01.txt"),
        ucf_root=UCF_ROOT,
        num_frames=NUM_FRAMES
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=partial(collate_fn, processor=processor),  
        pin_memory=True if torch.cuda.is_available() else False
    )

    optimizer = AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler()  # mixed precision — genuinely helps here (VRAM + speed)

    total_batches_in_epoch = len(train_loader)
    print(f"Total batches in one epoch: {total_batches_in_epoch}")
    print(f"Measuring the first {NUM_DEMO_BATCHES} batches to estimate full-epoch duration...\n")

    optimizer.zero_grad()
    start_time = time.perf_counter()
    measured_batches = 0

    for batch_idx, (pixel_values, labels) in enumerate(tqdm(train_loader, total=NUM_DEMO_BATCHES)):
        if batch_idx >= NUM_DEMO_BATCHES:
            break

        pixel_values = pixel_values.to(device)
        labels = labels.to(device)

        with autocast():
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()

        if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        measured_batches += 1

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    sec_per_batch = elapsed / measured_batches
    estimated_epoch_time_sec = sec_per_batch * total_batches_in_epoch
    estimated_epoch_time_min = estimated_epoch_time_sec / 60

    print("\n" + "=" * 50)
    print("VideoMAE Fine-tune Duration Estimate (Demo)")
    print("=" * 50)
    print(f"Measured batches:              {measured_batches}")
    print(f"Measured total time:           {elapsed:.1f} sec")
    print(f"Average time per batch:        {sec_per_batch:.2f} sec")
    print(f"Estimated time for 1 epoch:    {estimated_epoch_time_min:.1f} min")
    print(f"Estimated time for 5 epochs:   {estimated_epoch_time_min*5/60:.1f} hours")
    print(f"Estimated time for 100 epochs: {estimated_epoch_time_min*100/60:.1f} hours")
    print("=" * 50)


if __name__ == '__main__':
    main()  