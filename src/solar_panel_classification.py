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
data['t']=data['t']-1
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
        self.fc2 = nn.Linear(128, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 9)


    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu(self.fc3(x))
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

num_epochs=50

# Training starten
for epoch in range(num_epochs):
    correct = 0
    total=0
    for i, data in enumerate(trainloader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


        # Vorhersagen bestimmen
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print('Epoch {}/{}: Loss = {:.4f}, Accuracy = {:.4f}'.format(epoch+1, num_epochs, loss.item(), accuracy))
