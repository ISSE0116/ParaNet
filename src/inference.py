################################
#                              #
#          regression          #
#          Inference2          #
################################
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models
from torchvision.datasets.utils import download_url
import os
import sys

################################ hyper parameter #################################

inp_model_path = sys.argv[1]
inp_model = str(sys.argv[2])
img_dir = sys.argv[3]

weight_dir = '../weight_regression_path/' 
PATH = os.path.join(weight_dir, inp_model_path) 

###################################### model #####################################

def get_device(use_gpu):
    if use_gpu and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        return torch.device("cuda:2")
    else:
        return torch.device("cpu")

device = get_device(use_gpu=True)

if(inp_model == "vgg16"):
    model = models.vgg16(pretrained=True)
    num_ftrs = model.classifieri[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 1)
    model.load_state_dict(torch.load(PATH, map_location = device))
if(inp_model == "resnet18"):
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model.load_state_dict(torch.load(PATH, map_location = device))
if(inp_model == "resnext50_32x4d"):
    model = models.resnext50_32x4d(pretrained=True)    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model.load_state_dict(torch.load(PATH, map_location = device))

transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def _get_img_paths(img_dir):
    img_dir = Path(img_dir)
    img_extensions = [".jpg", ".jpeg", ".png"]
    img_paths = [str(p) for p in img_dir.iterdir() if p.suffix in img_extensions]
    img_paths.sort()

    return img_paths

class ImageFolder(Dataset):
    def __init__(self, img_dir, transform):
        self.img_paths = _get_img_paths(img_dir)
        self.transform = transform

    def __getitem__(self, index):
        path = self.img_paths[index]
        img = Image.open(path)
        inputs = self.transform(img)

        return {"image": inputs, "path": path}

    def __len__(self):
        return len(self.img_paths)

dataset = ImageFolder(img_dir, transform)
dataloader = DataLoader(dataset, batch_size=8)

##################################### inference ##################################

model.eval()

for batch in dataloader:
    inputs = batch["image"].to('cpu')
    outputs = model(inputs)
    
    for output, path in zip(outputs, batch["path"]):
        print("path: {}".format(os.path.basename(path)))
        print("output: {:.2f}".format(output.item()))
    
    print()
