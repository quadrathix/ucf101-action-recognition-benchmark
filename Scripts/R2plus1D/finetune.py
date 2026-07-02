import torch
import torchvision
import numpy as np
import os
from decord import VideoReader, cpu
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import R2Plus1D_18_Weights
from torch.optim import AdamW
from tqdm import tqdm
from torchvision import transforms
import time 

# =====================
# CONFIGURATION
# =====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UCF_ROOT = os.path.join(BASE_DIR, "data", "UCF101")
ANNOTATION_DIR = os.path.join(BASE_DIR, "data", "UCF101TrainTestSplits-RecognitionTask")
SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "r2plus1d_ucf101.pth")
NUM_EPOCHS = 5
BATCH_SIZE = 8       
NUM_WORKERS = 0        
NUM_FRAMES = 16


class UCF101Dataset(Dataset):
    def __init__(self, annotation_file, ucf_root, num_frames=16):
        self.ucf_root = ucf_root
        self.num_frames = num_frames
        self.label2id = {}
        self.samples = []
        self.resize_transform = transforms.Resize((112, 112))

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
                return torch.zeros((3, self.num_frames, 112, 112)), label

            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            frames = vr.get_batch(indices).asnumpy()  # (T, H, W, C), RGB

        except Exception:
            return torch.zeros((3, self.num_frames, 112, 112)), label

        video = torch.tensor(frames).permute(3, 0, 1, 2).float() / 255.0  # (C, T, H, W)
        video = self.resize_transform(video)

        mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
        std = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)
        video = (video - mean) / std

        return video, label


def main():
    print("Loading pretrained R(2+1)D model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = R2Plus1D_18_Weights.DEFAULT
    model = torchvision.models.video.r2plus1d_18(weights=weights)

    model.fc = torch.nn.Linear(model.fc.in_features, 101)
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

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        data_time_total = 0.0
        gpu_time_total = 0.0
        t_prev = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for batch_idx, (videos, labels) in enumerate(pbar):
            # DataLoader'dan batch'in bize ulaşana kadar geçen süre
            t_data_end = time.time()
            data_time = t_data_end - t_prev
            data_time_total += data_time

            videos = videos.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # GPU senkronizasyonu şart — yoksa CUDA asenkron çalıştığı için
            # ölçüm yanlış çıkar (işlem bitmeden zaman damgası alınır)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t_gpu_end = time.time()
            gpu_time = t_gpu_end - t_data_end
            gpu_time_total += gpu_time

            total_loss += loss.item()
            predicted = outputs.argmax(-1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            if batch_idx % 20 == 0:
                pbar.set_postfix({
                    'data_s': f"{data_time:.2f}",
                    'gpu_s': f"{gpu_time:.2f}"
                })

            t_prev = time.time()

        print(f"Epoch {epoch+1} — Avg Loss: {total_loss/len(train_loader):.4f} — Train Accuracy: {correct/total*100:.2f}%")
        print(f"  Toplam data-loading süresi: {data_time_total:.1f}s | Toplam GPU süresi: {gpu_time_total:.1f}s")
        print(f"  Oran (data/gpu): {data_time_total/max(gpu_time_total,1e-6):.2f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'label2id': train_dataset.label2id,
        'id2label': train_dataset.id2label
    }, SAVE_PATH)

    print(f"Model successfully saved to {SAVE_PATH}")


if __name__ == '__main__':
    main()