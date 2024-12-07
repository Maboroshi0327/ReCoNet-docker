import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from totaldata import *
from skimage import io, transform

data = ConsolidatedDataset(MPI_path="../datasets/MPI-Sintel-complete/", FC_path="../datasets/FlyingChairs2/")
print(len(data))
img1, img2, mask, flow = data[230]

# Check the data type and shape of the images
print(img1.size(), img2.size(), mask.size(), flow.size())
print(img1.dtype, img2.dtype)

# Convert th images to numpy arrays
img1 = (img1.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
img2 = (img2.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

io.imsave("img1.png", img1)
io.imsave("img2.png", img2)
