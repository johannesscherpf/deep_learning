import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np

# Check if CUDA (GPU support) is available
is_cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {is_cuda_available}")
print(torch.cuda.is_available())

data_dir = "../data/train_cnn"

dataset = ImageFolder(data_dir,transform = transforms.Compose([
       transforms.RandomHorizontalFlip(),
       transforms.RandomRotation(10),
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

train_loader = torch.utils.data.DataLoader(dataset, batch_size=256,
                                          shuffle=True)

print(dataset.classes)
img, label = dataset[0]
print(img.shape,label)


"""
dataiter = iter(trainloader)
images, labels = next(dataiter)
plt.imshow(np.transpose(torchvision.utils.make_grid(
    images[:25], normalize=True, padding=1, nrow=5).numpy(), (1, 2, 0)))
plt.axis('off')
plt.show()
"""

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32,
                            kernel_size=5, padding=1),
            torch.nn.Softplus(),
            torch.nn.Conv2d(in_channels=32, out_channels=32,
                            kernel_size=5, padding=1),
            torch.nn.Softplus(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Flatten(),
            torch.nn.Linear(6272, 512),
            torch.nn.Softplus(),
            torch.nn.Linear(512, 256),
            torch.nn.Linear(256, 9),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CNN().to(device)

num_epochs = 50
learning_rate = 0.001
weight_decay = 0.01
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay)


train_loss_list = []
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}:', end=' ')
    train_loss = 0
    correct = 0
    total = 0
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_accuracy = 100 * correct / total
    train_loss_list.append(train_loss / len(train_loader))
    print(f"Training loss = {train_loss_list[-1]}" f" Training Acc: {train_accuracy:.2f}% ")