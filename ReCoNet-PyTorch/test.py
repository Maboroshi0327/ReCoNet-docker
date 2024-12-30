import torch

from datasets import Coco2014, toPil
from network import ReCoNet

data = Coco2014("C:\\Datasets\\coco2014")
img = data[100].unsqueeze(0)

model = ReCoNet()
model.load_state_dict(torch.load("./models/Coco2014_epoch_2_batchSize_4.pth", weights_only=True))

_, output = model(img)
img = toPil(img[0].byte())
img.show()
outimg = toPil(output[0].byte())
outimg.show()
