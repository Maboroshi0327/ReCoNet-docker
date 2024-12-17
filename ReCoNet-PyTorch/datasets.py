import os
from typing import Union

import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFilter

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset

from flowlib import read


def list_files(directory):
    files = [f.path for f in os.scandir(directory) if f.is_file()]
    return sorted(files)


def visualize_flow(flow):
    hsv = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[0].cpu().numpy(), flow[1].cpu().numpy())
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return rgb


def warp(x, flo, padding_mode="zeros"):
    B, C, H, W = x.size()

    # Mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo

    # Scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid, mode="nearest", padding_mode=padding_mode, align_corners=False)
    return output


class FlyingThings3D(Dataset):
    def __init__(self, path: str, train: bool = True, resolution: tuple = (640, 360)):
        """
        path -> Path to the location where the "frames_finalpass", "optical_flow" and "motion_boundaries" folders are kept inside the FlyingThings3D folder. \\
        train -> True if training dataset is required, False if testing dataset is required. \\
        resolution -> Resolution of the images to be returned. Width first, then height.
        """
        path_frame = path + "frames_finalpass/" + ("TRAIN/" if train else "TEST/")
        path_flow = path + "optical_flow/" + ("TRAIN/" if train else "TEST/")
        path_motion = path + "motion_boundaries/" + ("TRAIN/" if train else "TEST/")

        assert os.path.exists(path_frame), f"Path {path_frame} does not exist."
        assert os.path.exists(path_flow), f"Path {path_flow} does not exist."
        assert os.path.exists(path_motion), f"Path {path_motion} does not exist."
        assert train in [True, False], "train must be a boolean."
        assert len(resolution) == 2 and isinstance(resolution, tuple), "Resolution must be a tuple of 2 integers."

        self.path = path
        self.frame = list()
        self.flow = list()
        self.motion = list()

        # progress bar
        pbar = tqdm(desc="Initial FlyingThings3D", total=20151 * 3)

        # frames_finalpass
        for abcpath in ["A/", "B/", "C/"]:
            for folder in os.listdir(path_frame + abcpath):
                files = list_files(path_frame + abcpath + folder + "/left/")
                for i in range(9):
                    self.frame.append((files[i], files[i + 1]))
                    pbar.update(1)
                break

        # optical_flow
        for abcpath in ["A/", "B/", "C/"]:
            for folder in os.listdir(path_flow + abcpath):
                into_future_files = list_files(path_flow + abcpath + folder + "/into_future/left/")
                into_past_files = list_files(path_flow + abcpath + folder + "/into_past/left/")
                for i in range(9):
                    self.flow.append(into_past_files[i + 1])
                    pbar.update(1)
                break

        # mask
        for abcpath in ["A/", "B/", "C/"]:
            for folder in os.listdir(path_motion + abcpath):
                files = list_files(path_motion + abcpath + folder + "/into_future/left/")
                for i in range(9):
                    self.motion.append(files[i + 1])
                    pbar.update(1)
                break

        self.length = len(self.frame)
        self.resolution = resolution

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> Union[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        idx -> Index of the image pair, optical flow and mask to be returned.
        """
        # convert to tensor
        toTensor = transforms.ToTensor()
        gaussianBlur = transforms.GaussianBlur(kernel_size=3, sigma=1.0)

        # read image
        img_path = self.frame[idx]
        img1 = Image.open(img_path[0]).convert("RGB").resize(self.resolution, Image.BILINEAR)
        img2 = Image.open(img_path[1]).convert("RGB").resize(self.resolution, Image.BILINEAR)
        img1 = toTensor(img1)
        img2 = toTensor(img2)

        # read flow
        flow_into_past = toTensor(read(self.flow[idx]).copy())[:-1]
        originalflowshape = flow_into_past.shape

        flow_into_past = F.interpolate(
            flow_into_past.unsqueeze(0),
            size=(self.resolution[1], self.resolution[0]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        flow_into_past[0] *= flow_into_past.shape[1] / originalflowshape[1]
        flow_into_past[1] *= flow_into_past.shape[2] / originalflowshape[2]

        # read motion boundaries
        motion = Image.open(self.motion[idx]).resize(self.resolution, Image.BILINEAR)
        motion = toTensor(motion).squeeze(0)
        motion[motion != 0] = 1
        motion = 1 - motion

        # create mask
        img_warp = warp(img1.unsqueeze(0), flow_into_past.unsqueeze(0)).squeeze(0)
        img_warp_blur = gaussianBlur(img_warp)
        img2_blur = gaussianBlur(img2)

        warp_error = torch.abs(img_warp_blur - img2_blur)
        warp_error = torch.sum(warp_error, dim=0)

        mask = warp_error < 0.1
        mask = mask.float()
        mask = mask * motion

        return img1, img2, flow_into_past, mask


if __name__ == "__main__":
    # Test FlyingThings3D
    fly = FlyingThings3D("/root/datasets/flyingthings3d/", train=True)

    pbar = tqdm(range(9), desc="Test FlyingThings3D", leave=True)
    for i in pbar:
        img1, img2, flow_into_past, mask = fly.__getitem__(i * 3)

        # warp image & visualize flow
        next_img = warp(img1.unsqueeze(0), flow_into_past.unsqueeze(0)).squeeze(0)
        warp_mask = mask * next_img
        flow_rgb = visualize_flow(flow_into_past)

        # convert to PIL
        to_pil = transforms.ToPILImage()
        img1 = to_pil(img1)
        img2 = to_pil(img2)
        mask = to_pil(mask)
        next_img = to_pil(next_img)
        warp_mask = to_pil(warp_mask)

        # delete old images
        files = list_files(f"./datasets_images/{i + 1}/")
        for f in files:
            os.remove(f)

        # save images
        img1.save(f"./datasets_images/{i + 1}/img1.png")
        img2.save(f"./datasets_images/{i + 1}/img2.png")
        # mask.save(f"./datasets_images/{i + 1}/mask.png")
        next_img.save(f"./datasets_images/{i + 1}/next_img.png")
        warp_mask.save(f"./datasets_images/{i + 1}/warp_mask.png")
        # cv2.imwrite(f"./datasets_images/{i + 1}/flow.png", flow_rgb)
