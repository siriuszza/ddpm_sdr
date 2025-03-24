import sys
import os
os.chdir("F:\\office\\research\\diff_dim_reduc\\ddpm_sdr")
# os.getcwd()
sys.path.append(os.path.abspath(os.path.dirname("modules.py")))
from utils import *
import torch
from torchvision.utils import save_image

from ddpm_conditional import Diffusion
from utils import get_data
import argparse
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets
import torchvision
import torchvision.transforms as T

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.batch_size = 64  # 5
args.img_size = 28
# args.dataset_path = r"C:\Users\dome\datasets\landscape_img_folder"


# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

def get_data(args):
    train_transforms = torchvision.transforms.Compose([
        T.Resize(args.img_size + int(.25*args.img_size)),  # args.img_size + 1/4 *args.img_size
        T.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    val_transforms = torchvision.transforms.Compose([
        T.Resize(args.img_size),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transforms)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, val_loader = get_data(args)
#diff = Diffusion(device="cpu")
diff = Diffusion(device=device) 

image = next(iter(train_loader))[0][0].to(device)
t = torch.Tensor([50, 100, 150, 200, 300, 600, 700, 999]).long().to(device)

noised_image, _ = diff.noise_images(image, t)
save_image(noised_image.add(1).mul(0.5), "noise.jpg")


