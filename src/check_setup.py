
import torch
import numpy as np
import pandas as pd
import matplotlib
import cv2
from PIL import Image

print("=" * 50)
print("LESION SEGMENTATION PROJECT")
print("=" * 50)

print("PyTorch:", torch.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("OpenCV:", cv2.__version__)

device = torch.device("cpu")

print("Selected device:", device)

x = torch.rand(1000, 1000, device=device)
y = x @ x

print("Tensor calculation:", y.shape)
print("Setup successful!")
