import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

from utilities import *
from network import *
from totaldata import *

LR = 1e-3
epochs = 100
device = "cuda"

batch_size = 4
dataloader = DataLoader(FlyingChairsDataset("../datasets/FlyingChairs2/"), batch_size=batch_size)
model = ReCoNet().to(device)

resume = input("Resume training? y/n: ").lower() == "y"
if resume:
    model_name = input("Model Name: ")
    model.load_state_dict(torch.load("runs/output/" + model_name))

adam = optim.Adam(model.parameters(), lr=LR)
L2distance = nn.MSELoss().to(device)
L2distancematrix = nn.MSELoss(reduction="none").to(device)
Vgg16 = Vgg16().to(device)

style_names = ("autoportrait", "candy", "composition", "edtaonisl", "udnie")
style_model_path = "./models/weights/"
style_img_path = "./models/style/" + style_names[2]
style = transform1(Image.open(style_img_path + ".jpg"))
style = style.unsqueeze(0).expand(batch_size, 3, IMG_SIZE[0], IMG_SIZE[1]).to(device)

for param in Vgg16.parameters():
    param.requires_grad = False

STYLE_WEIGHTS = [1e-1, 1e0, 1e1, 5e0]
styled_featuresR = Vgg16(normalize(style))
style_GM = [gram_matrix(f) for f in styled_featuresR]

for epoch in range(epochs):
    batch_iterator = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", leave=False)
    for itr, (img1, img2, mask, flow) in enumerate(batch_iterator):
        img1 = img1.to(device)
        img2 = img2.to(device)
        mask = mask.to(device)
        flow = -flow.to(device)
        adam.zero_grad()

        if (itr + 1) % 500 == 0:
            for param in adam.param_groups:
                param["lr"] = max(param["lr"] / 1.2, 1e-4)

        feature_map1, styled_img1 = model(img1)
        feature_map2, styled_img2 = model(img2)
        styled_img1 = normalize(styled_img1)
        styled_img2 = normalize(styled_img2)
        img1, img2 = normalize(img1), normalize(img2)

        styled_features1 = Vgg16(styled_img1)
        styled_features2 = Vgg16(styled_img2)
        img_features1 = Vgg16(img1)
        img_features2 = Vgg16(img2)

        feature_flow = nn.functional.interpolate(flow, size=feature_map1.shape[2:], mode="bilinear")
        feature_flow[:, 0, :, :] *= float(feature_map1.shape[2]) / flow.shape[2]
        feature_flow[:, 1, :, :] *= float(feature_map1.shape[3]) / flow.shape[3]

        feature_mask = nn.functional.interpolate(
            mask.view(batch_size, 1, 640, 360), size=feature_map1.shape[2:], mode="bilinear"
        )
        warped_fmap = warp(feature_map1, feature_flow)

        f_temporal_loss = torch.sum(feature_mask * L2distancematrix(feature_map2, warped_fmap))
        f_temporal_loss *= LAMBDA_F / (feature_map2.numel())

        warped_style = warp(styled_img1, flow)
        warped_image = warp(img1, flow)

        output_term = styled_img2 - warped_style
        input_term = img2 - warped_image

        input_term = 0.2126 * input_term[:, 0, :, :] + 0.7152 * input_term[:, 1, :, :] + 0.0722 * input_term[:, 2, :, :]
        input_term = input_term.unsqueeze(1).expand(-1, 3, -1, -1)

        mask = mask.unsqueeze(1)
        o_temporal_loss = torch.sum(mask * L2distancematrix(output_term, input_term))
        o_temporal_loss *= LAMBDA_O / (img1.numel())

        content_loss = 0
        content_loss += L2distance(styled_features1[2], img_features1[2])
        content_loss += L2distance(styled_features2[2], img_features2[2])
        content_loss *= ALPHA / (styled_features1[2].numel())

        style_loss = 0
        for i, weight in enumerate(STYLE_WEIGHTS):
            gram_s = style_GM[i]
            gram_img1 = gram_matrix(styled_features1[i])
            gram_img2 = gram_matrix(styled_features2[i])
            style_loss += weight * (L2distance(gram_img1, gram_s) + L2distance(gram_img2, gram_s))
        style_loss *= BETA

        reg_loss = GAMMA * (
            torch.sum(torch.abs(styled_img1[:, :, :, :-1] - styled_img1[:, :, :, 1:]))
            + torch.sum(torch.abs(styled_img1[:, :, :-1, :] - styled_img1[:, :, 1:, :]))
            + torch.sum(torch.abs(styled_img2[:, :, :, :-1] - styled_img2[:, :, :, 1:]))
            + torch.sum(torch.abs(styled_img2[:, :, :-1, :] - styled_img2[:, :, 1:, :]))
        )

        loss = f_temporal_loss + o_temporal_loss + content_loss + style_loss + reg_loss
        loss.backward()
        adam.step()

        batch_iterator.set_postfix(
            SL=style_loss.item(),
            CL=content_loss.item(),
            FTL=f_temporal_loss.item(),
            OTL=o_temporal_loss.item(),
            RL=reg_loss.item(),
        )

    torch.save(model.state_dict(), "%s/reconet_batch_%d_epoch_%d.pth" % ("runs/output", batch_size, epoch))
