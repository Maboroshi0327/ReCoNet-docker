import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from PIL import Image
from tqdm import tqdm

from datasets import FlyingThings3D
from network import ReCoNet, Vgg16


device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = 100
LR = 1e-3
ALPHA = 1e13
BETA = 1e10
GAMMA = 3e-2
LAMBDA_O = 1e6
LAMBDA_F = 1e4
STYLE_WEIGHTS = [1e-1, 1e0, 1e1, 5e0]
IMG_SIZE = (640, 360)


def gram_matrix(y: torch.Tensor):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def vgg_normalize(batch: torch.Tensor):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch - mean) / std


def train():
    # Datasets and model
    # dataloader = DataLoader(FlyingThings3D("../datasets/flyingthings3d/"), batch_size=2)
    model = ReCoNet().to(device)

    # Optimizer and loss
    adam = optim.Adam(model.parameters(), lr=LR)
    L2distance = nn.MSELoss().to(device)
    L2distancematrix = nn.MSELoss(reduction="none").to(device)
    vgg16 = Vgg16().to(device)
    for param in vgg16.parameters():
        param.requires_grad = False

    # Style image
    toTensor = transforms.ToTensor()
    style_names = ("autoportrait", "candy", "composition", "edtaonisl", "udnie")
    style_img_path = "./styles/" + style_names[2] + ".jpg"
    style = Image.open(style_img_path).convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
    style = toTensor(style).unsqueeze(0).to(device)

    # Style image Gram Matrix
    style_features = vgg16(vgg_normalize(style))
    style_GM = [gram_matrix(f) for f in style_features]


if __name__ == "__main__":
    train()
