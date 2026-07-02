# ucf101-action-recognition-benchmark
A comparison of I3D, R2plus1D and VideoMAE on UCF101

## Before running any script:

1. Install the base requirements
pip install -r requirements.txt

2. Install PyTorch with CUDA 12.4 support
Requires an NVIDIA GPU with a driver compatible with CUDA 12.4 (CUDA Toolkit itself is NOT required — the wheel bundles it)
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124

3. Data Setup
This repository does not include the UCF101 dataset (too large for GitHub), therefore it needs to be downloaded separately.

1- Get a Kaggle API key from https://www.kaggle.com/settings → API → "Create New Token" (downloads `kaggle.json` with your username and key).
2- Open `Scripts/import_ucf.py` and fill in your `username` and `key` in place of the `[USERNAME]` / `[KAGGLE KEY]` placeholders.
3- Run the script: python Scripts/import_ucf.py
4- Download and extract the UCF101 dataset into `data/` (matching the folder structure expected by the training/eval scripts — one subfolder per action class, e.g. `data/UCF101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi`).
