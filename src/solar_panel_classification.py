import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Daten laden
data = pd.read_csv('../data/train.csv')
data = data.dropna()

# Daten vorverarbeiten
data_X = data.drop(columns = ['t'])
data_X = (data_X - data_X.mean()) / data_X.std()
data_X = torch.from_numpy(data_X.values).float()
data_Y = data['t']
data_Y = torch.as_tensor(data_Y,dtype=torch.long)

# Train-Test-Split

# Netzwerkarchitektur definieren
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(60, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 9)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return x

# Training
# Parameter festlegen
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)
batch_size = 32

# Dataset laden
Dataset=TensorDataset(data_X, data_Y)
trainloader = DataLoader(Dataset, batch_size=batch_size, shuffle=True)

num_epochs=10

# Training starten
for epoch in range(num_epochs):
    correct = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        correct += (outputs == labels).float().sum()
        accuracy = 100 * correct / len(data_X)
    print('Epoch {}/{}: Loss = {:.4f}, Accuracy = {:.4f}'.format(epoch+1, num_epochs, loss.item(), accuracy))

