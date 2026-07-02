import torch
import numpy as np
from decord import VideoReader, cpu
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from tqdm import tqdm
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UCF_ROOT = os.path.join(BASE_DIR, "data", "UCF101")
ANNOTATION_DIR = os.path.join(BASE_DIR, "data", "UCF101TrainTestSplits-RecognitionTask")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = VideoMAEImageProcessor.from_pretrained("nateraw/videomae-base-finetuned-ucf101")
model = VideoMAEForVideoClassification.from_pretrained("nateraw/videomae-base-finetuned-ucf101")
model = model.to(device)
model.eval()
print(f"Model loaded — Classes: {model.config.num_labels}")

annotation_path = os.path.join(ANNOTATION_DIR, "testlist01.txt")
with open(annotation_path, "r") as f:
    test_videos = [line.strip() for line in f.readlines() if line.strip()]
print(f"Test videos: {len(test_videos)}")

# Model kartındaki resmi örnekleme fonksiyonu — birebir aynı mantık
def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    if converted_len >= seg_len:
        # Video, istenen pencereden kısaysa elimizdeki tüm kareleri kullan
        indices = np.linspace(0, seg_len - 1, num=clip_len)
        return np.clip(indices, 0, seg_len - 1).astype(np.int64)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

def get_video_frames(video_path, num_frames=16, frame_sample_rate=4):
    vr = VideoReader(video_path, ctx=cpu(0))
    seg_len = len(vr)
    if seg_len < 1:
        return None
    indices = sample_frame_indices(clip_len=num_frames, frame_sample_rate=frame_sample_rate, seg_len=seg_len)
    video = vr.get_batch(indices).asnumpy()  # (T, H, W, C), RGB
    return list(video)  # processor liste bekliyor

correct = 0
total = 0
errors = 0

np.random.seed(0)  # tekrar üretilebilirlik için — model kartı da bunu kullanıyor

for video_rel_path in tqdm(test_videos):
    try:
        true_label = video_rel_path.split("/")[0]
        video_path = os.path.join(UCF_ROOT, video_rel_path)

        frames = get_video_frames(video_path)
        if frames is None or len(frames) < 16:
            errors += 1
            continue

        inputs = processor(frames, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        predicted_idx = outputs.logits.argmax(-1).item()
        predicted_label = model.config.id2label[predicted_idx]

        if true_label.lower() == predicted_label.lower().replace(" ", ""):
            correct += 1
        total += 1

    except Exception as e:
        errors += 1
        continue

print(f"Total videos evaluated: {total}")
print(f"Correct predictions: {correct}")
print(f"Accuracy: {correct/total*100:.2f}%" if total > 0 else "N/A")
print(f"Errors/skipped: {errors}")