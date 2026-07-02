import torch
from decord import VideoReader, cpu
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import numpy as np
from tqdm import tqdm
import os

# =====================
# CONFIGURATION
# =====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UCF_ROOT = os.path.join(BASE_DIR, "data", "UCF101")
ANNOTATION_DIR = os.path.join(BASE_DIR, "data", "UCF101TrainTestSplits-RecognitionTask")
CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videomae_ucf101_epoch4.pth")  # epoch4/epoch5 arasında değiştir
NUM_FRAMES = 16
# =====================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fine-tune edilmiş checkpoint'i yükle
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
id2label = checkpoint['id2label']
label2id = checkpoint['label2id']
num_classes = len(id2label)

processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
model = VideoMAEForVideoClassification.from_pretrained(
    "MCG-NJU/videomae-base",
    num_labels=num_classes,
    ignore_mismatched_sizes=True
)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()
print(f"Model loaded from {CHECKPOINT_PATH} — Classes: {num_classes}")

annotation_path = os.path.join(ANNOTATION_DIR, "testlist01.txt")
with open(annotation_path, "r") as f:
    test_videos = [line.strip() for line in f.readlines() if line.strip()]
print(f"Test videos: {len(test_videos)}")

def get_video_frames(video_path, num_frames=NUM_FRAMES):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    if total_frames < num_frames:
        return None
    # Fine-tune ile tutarlı: videonun tamamına yayılmış uniform sampling (nateraw'ın segment-tabanlı protokolü DEĞİL)
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()  # (T, H, W, C)
    return list(frames)

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
        frames = get_video_frames(video_path, NUM_FRAMES)
        if frames is None:
            errors += 1
            continue

        inputs = processor(frames, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            predicted_id = outputs.logits.argmax(-1).item()

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