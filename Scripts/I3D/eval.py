import torch
from decord import VideoReader, cpu
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UCF_ROOT = os.path.join(BASE_DIR, "data", "UCF101")
ANNOTATION_DIR = os.path.join(BASE_DIR, "data", "UCF101TrainTestSplits-RecognitionTask")
CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "i3d_ucf101.pth")
NUM_FRAMES = 8       # I3D fine-tune'da kullanılan konfigürasyonla tutarlı
CROP_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fine-tune edilmiş checkpoint'i yükle
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
id2label = checkpoint['id2label']
label2id = checkpoint['label2id']
num_classes = len(id2label)

model = torch.hub.load("facebookresearch/pytorchvideo", "i3d_r50", pretrained=False)
model.blocks[6].proj = torch.nn.Linear(2048, num_classes)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

annotation_path = os.path.join(ANNOTATION_DIR, "testlist01.txt")
with open(annotation_path, "r") as f:
    test_videos = [line.strip() for line in f.readlines() if line.strip()]
print(f"Test videos: {len(test_videos)}")

resize_transform = transforms.Resize((CROP_SIZE, CROP_SIZE))
mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)

def get_video_tensor(video_path, num_frames=NUM_FRAMES):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    if total_frames < num_frames:
        return None
    # Fine-tune'da kullandığımızla tutarlı: videonun tamamına yayılmış uniform sampling
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()  # (T, H, W, C)

    video = torch.tensor(frames).permute(3, 0, 1, 2).float() / 255.0  # (C, T, H, W)
    video = resize_transform(video)
    video = (video - mean) / std
    return video.unsqueeze(0)  # (1, C, T, H, W)

correct = 0
total = 0
errors = 0

for video_rel_path in tqdm(test_videos):
    try:
        true_label_name = video_rel_path.split("/")[0]
        if true_label_name not in label2id:
            errors += 1
            continue
        true_label_id = label2id[true_label_name]

        video_path = os.path.join(UCF_ROOT, video_rel_path)
        video_tensor = get_video_tensor(video_path, NUM_FRAMES)
        if video_tensor is None:
            errors += 1
            continue

        video_tensor = video_tensor.to(device)

        with torch.no_grad():
            outputs = model(video_tensor)
            predicted_id = outputs.argmax(-1).item()

        if predicted_id == true_label_id:
            correct += 1
        total += 1

    except Exception as e:
        errors += 1
        continue

print(f"Total videos evaluated: {total}")
print(f"Correct predictions: {correct}")
print(f"Accuracy: {correct/total*100:.2f}%" if total > 0 else "Accuracy: N/A (total=0)")
print(f"Errors/skipped: {errors}")