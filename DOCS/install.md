# Step-by-step installation instructions


**1. Create a conda virtual environment and activate it.**
```shell
conda create -n drivedreamer python=3.10 -y
conda activate drivedreamer
```

**2. Install PyTorch and torchvision (tested on torch>=2.2 & cuda=12.1).**
```shell
pip install torch torchvision
```

**3. Install other dependencies.**
```shell
pip install nuscenes-devkit
pip install lmdb
pip install decord
pip install accelerate
pip install transformers
pip install pytorch-fid
pip install lpips
pip install terminaltables
pip install opencv-python
pip install ftfy
pip install einops
pip install compel
pip install bs4
pip install open_clip_torch
pip install pynvml
pip install paramiko
pip install tensorboard
pip install mediapipe
pip install deepspeed
pip install diffusers
pip install imageio
pip install imageio[ffmpeg]
pip install scikit-image
```
