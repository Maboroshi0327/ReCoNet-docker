import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset


def list_files(directory):
    files = [f.path for f in os.scandir(directory) if f.is_file()]
    return sorted(files)


class FlyingThings3D(Dataset):
    def __init__(self, path, train=True, resolution=(640, 360)):
        self.path = path
        self.each_data = list()

        path = path + "frames_finalpass/" + ("TRAIN/" if train else "TEST/")
        for abcpath in ["A/", "B/", "C/"]:
            for folder in os.listdir(path + abcpath):
                files = list_files(path + abcpath + folder + "/left/")
                for i in range(9):
                    self.each_data.append((files[i], files[i + 1], abcpath + folder))

        self.length = len(self.each_data)
        self.resolution = resolution

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img_path = self.each_data[idx]
        img1 = Image.open(img_path[0]).convert("RGB").resize(self.resolution)
        img2 = Image.open(img_path[1]).convert("RGB").resize(self.resolution)

        # 定義轉換
        transform = transforms.ToTensor()

        # 將 PIL 圖像轉換為張量
        img1 = transform(img1)
        img2 = transform(img2)

        return img1, img2


if __name__ == "__main__":
    fly = FlyingThings3D("/root/datasets/flyingthings3d/", train=True)
    fly.__getitem__(0)
