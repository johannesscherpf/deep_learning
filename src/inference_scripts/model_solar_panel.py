import torch
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim


# --- gleiche Architektur wie beim Training ---
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(60, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256, 9)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class Model:
    def __init__(self):
        # Gesamtes Modell laden
        script_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(script_dir, 'solar_weights.pth')
        mean_path = os.path.join(script_dir, 'mean.npy')
        std_path = os.path.join(script_dir, 'std.npy')

        self.net = Net()
        self.net.load_state_dict(torch.load(model_path, map_location="cpu"))
        #self.net = torch.load(model_path, map_location="cpu", weights_only=False)
        #self.net = torch.load("solar.pth", map_location="cpu")
        self.net.eval()

        self.mean = np.load(mean_path)
        self.std = np.load(std_path)

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        # In Tensor konvertieren
        inputs = (x_test - self.mean) / self.std
        inputs = torch.from_numpy(inputs.astype(np.float32))

        with torch.no_grad():
            outputs = self.net(inputs)
            _, predicted = torch.max(outputs, dim=1)
        return predicted.numpy() + 1