import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import os


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
            nn.MaxPool2d(2, 2),  # 16x16

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 8x8

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 4x4

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


class Model:

    def __init__(self):
        self.categories = ["airplane", "automobile", "bird", "cat",
                           "deer", "dog", "horse", "ship", "truck"]

        # Modell laden
        script_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(script_dir, 'cnn.pth')

        self.CNN = CNN()  # <-- Instanz
        state_dict = torch.load(model_path, map_location="cpu")
        self.CNN.load_state_dict(state_dict)
        self.CNN.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4936, 0.4869, 0.4577),
                                 (0.2030, 0.2018, 0.2049))
        ])

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        # Bild transformieren
        img = self.transform(x_test)

        # Batch dimension hinzufügen: (1, 3, H, W)
        img = img.unsqueeze(0)

        with torch.no_grad():
            outputs = self.CNN(img)
            predicted = outputs.cpu().numpy()

        return predicted