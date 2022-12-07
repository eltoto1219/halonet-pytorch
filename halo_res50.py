from time import time

import torch
from torch import nn
from torchvision.models import resnet50

from bottleneck_transformer_pytorch import BottleStack
from halonet_pytorch import HaloStack

layer = HaloStack(
    dim=256,
    fmap_size=56,  # set specifically for imagenet's 224 x 224
    dim_out=2048,
    proj_factor=4,
    downsample=True,
    heads=4,
    dim_head=128,
    rel_pos_emb=True,
    activation=nn.ReLU(),
)

resnet = resnet50()

# model surgery

backbone = list(resnet.children())

model = nn.Sequential(
    *backbone[:5],
    layer,
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(1),
    nn.Linear(2048, 1000)
)

model = model.cuda()


# use the 'BotNet'

img = torch.randn(1, 3, 224, 224).cuda()
s = time()
preds = model(img)  # (2, 1000)
e = time()

print(e - s)
