# %%
import torch
from decord import VideoReader, cpu
from torchvision import transforms
import numpy as np
import time
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UCF_ROOT = os.path.join(BASE_DIR, "data", "UCF101")
CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "i3d_ucf101.pth")
NUM_FRAMES = 8
CROP_SIZE = 224

# %%
print("Loading fine-tuned I3D (i3d_r50) model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
num_classes = len(checkpoint['id2label'])

model = torch.hub.load("facebookresearch/pytorchvideo", "i3d_r50", pretrained=False)
model.blocks[6].proj = torch.nn.Linear(2048, num_classes)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print("Model loaded")

# %%
# Num of parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model size: {total_params * 4 / 1024**2:.2f} MB")  # float32 = 4 byte

# %%
# Example video — eval/train script'iyle tutarlı örnekleme (decord + linspace)
video_path = os.path.join(UCF_ROOT, "Basketball", "v_Basketball_g01_c01.avi")

vr = VideoReader(video_path, ctx=cpu(0))
total_frames = len(vr)
indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)
frames = vr.get_batch(indices).asnumpy()  # (T, H, W, C)

video = torch.tensor(frames).permute(3, 0, 1, 2).float() / 255.0  # (C, T, H, W)
resize_transform = transforms.Resize((CROP_SIZE, CROP_SIZE))
video = resize_transform(video)

mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)
video = (video - mean) / std

pixel_values = video.unsqueeze(0).to(device)  # (1, C, T, H, W)

print(f"Input shape: {pixel_values.shape}")

# %%
# Warm-up (10 run)
for _ in range(10):
    with torch.no_grad():
        _ = model(pixel_values)
torch.cuda.synchronize()

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

start = time.perf_counter()
with torch.no_grad():
    outputs = model(pixel_values)
torch.cuda.synchronize()
end = time.perf_counter()

inference_time = (end - start) * 1000  # ms
peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB

print(f"Inference time: {inference_time:.2f} ms")
print(f"Peak GPU Memory: {peak_memory:.2f} MB")

# %%
from fvcore.nn import FlopCountAnalysis

flops = FlopCountAnalysis(model, pixel_values)
flops.unsupported_ops_warnings(False)
flops.uncalled_modules_warnings(False)

total_flops = flops.total()
print(f"FLOPs: {total_flops:,}")
print(f"GFLOPs: {total_flops / 1e9:.2f}")

# %%
print("=" * 45)
print("I3D Complexity Report")
print("=" * 45)
print(f"Total Parameters:      {total_params:,}")
print(f"Trainable Param:       {trainable_params:,}")
print(f"Model Size:            {total_params * 4 / 1024**2:.2f} MB")
print(f"GFLOPs (fvcore):       {total_flops / 1e9:.2f}")
print(f"Inference Time:        {inference_time:.2f} ms")
print(f"Peak GPU Memory:       {peak_memory:.2f} MB")
print(f"Test Environment:      NVIDIA RTX 4060 Laptop GPU")
print("Note: fvcore may undercount ops for some 3D-conv/pooling")
print("      layers depending on operator support.")
print("=" * 45)