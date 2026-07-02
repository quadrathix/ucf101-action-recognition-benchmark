# ucf101-action-recognition-benchmark
A comparison of I3D, R2plus1D and VideoMAE on UCF101

## Before running any script:

1. Install the base requirements
pip install -r requirements.txt

2. Install PyTorch with CUDA 12.4 support
Requires an NVIDIA GPU with a driver compatible with CUDA 12.4 (CUDA Toolkit itself is NOT required — the wheel bundles it)
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
