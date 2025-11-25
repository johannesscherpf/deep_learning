import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim

transform = transforms.Compose([
    transforms.ToTensor()
])

data_dir = r"C:\Users\stjssche\Desktop\train"
dataset = ImageFolder(data_dir, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

def calculate_mean_std (train_loader):
    mean = 0
    std = 0
    total_images = 0

    for images, _ in train_loader:
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            total_images += batch_samples
    mean /= total_images
    std /= total_images

    return mean, std

mean, std = calculate_mean_std(train_loader)

print(mean, std)