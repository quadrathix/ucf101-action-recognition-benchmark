# %%
import torch
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import cv2
import numpy as np
import time
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UCF_ROOT = os.path.join(BASE_DIR, "data", "UCF101")
# %%
processor = VideoMAEImageProcessor.from_pretrained("nateraw/videomae-base-finetuned-ucf101")
model = VideoMAEForVideoClassification.from_pretrained("nateraw/videomae-base-finetuned-ucf101")

model = model.to("cuda")
model.eval()

print("Model downloaded")

# %%
# Num of parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model size: {total_params * 4 / 1024**2:.2f} MB")  # float32 = 4 byte

# %%
# Example video 
video_path = os.path.join(UCF_ROOT, "Basketball", "v_Basketball_g01_c01.avi")
cap = cv2.VideoCapture(video_path)
frames = []
while len(frames) < 16:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame)
cap.release()

inputs = processor(frames, return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}

print(f"Input shape: {inputs['pixel_values'].shape}")

# %%
# Warm-up (10 run)
for _ in range(10):
    with torch.no_grad():
        _ = model(**inputs)
torch.cuda.synchronize()

# Cleaning cache
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

start = time.perf_counter()
with torch.no_grad():
    outputs = model(**inputs)
torch.cuda.synchronize()
end = time.perf_counter()

inference_time = (end - start) * 1000  # ms
peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB

print(f"Inference time: {inference_time:.2f} ms")
print(f"Peak GPU Memory: {peak_memory:.2f} MB")

# %%
from fvcore.nn import FlopCountAnalysis

pixel_values = inputs['pixel_values']  # dict -> tensor

flops = FlopCountAnalysis(model, pixel_values)
flops.unsupported_ops_warnings(False)
flops.uncalled_modules_warnings(False)

total_flops = flops.total()
print(f"FLOPs: {total_flops:,}")
print(f"GFLOPs: {total_flops / 1e9:.2f}")

# %%
print("=" * 45)
print("VideoMAE Complexity Report")
print("=" * 45)
print(f"Total Parameters:      {total_params:,}")
print(f"Trainable Param:       {trainable_params:,}")
print(f"Model Size:            {total_params * 4 / 1024**2:.2f} MB")
print(f"GFLOPs (fvcore):       {total_flops / 1e9:.2f}")
print(f"GFLOPs (paper):        180.00")
print(f"Inference Time:        {inference_time:.2f} ms")
print(f"Peak GPU Memory:       {peak_memory:.2f} MB")
print(f"Test Environment:      NVIDIA RTX 4060 Laptop GPU")
print("=" * 45)
print("Note: fvcore calculates the transformer operations")
print("less then reality (softmax, LayerNorm, GELU)")
print("especially for transformer based models.")
print("=" * 45)


