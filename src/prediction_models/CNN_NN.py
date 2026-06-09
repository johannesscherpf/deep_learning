import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim

# Check if CUDA (GPU support) is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"CUDA Available: {torch.cuda.is_available()}")

# Data augmentation and transformation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4936, 0.4869, 0.4577), (0.2030, 0.2018, 0.2049))
])

transform_val = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.4936, 0.4869, 0.4577), (0.2030, 0.2018, 0.2049))])

data_dir = r"C:\Users\stjssche\Desktop\train"
dataset = ImageFolder(data_dir, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

val_data_dir = r"C:\Users\stjssche\Desktop\val"
val_dataset = ImageFolder(val_data_dir, transform=transform)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=128)


# CNN
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 16x16

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 8x8

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 4x4
            nn.Flatten()
        )
        self.fc_model = nn.Sequential(
            nn.Linear(512 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 9)
        )

    def forward(self, x):
        x = self.conv_model(x)
        x = self.fc_model(x)
        return x


model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def calculatevalloss(model, val_dataloader):
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    val_loss /= len(val_dataloader)
    val_accuracy = val_correct / val_total
    return val_loss, val_accuracy




# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}:', end=' ')
    train_loss = 0
    correct = 0
    total = 0
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    val_loss, val_accuracy = calculatevalloss(model, val_dataloader)
    train_accuracy = 100 * correct / total
    print(f"Training loss = {train_loss / len(train_loader):.4f} | Training Acc: {train_accuracy:.2f}% | Val_loss: {val_loss:.4f} | Val_Acc: {val_accuracy:.2f}")

torch.save(model.state_dict(), '../models/cnn.pth')