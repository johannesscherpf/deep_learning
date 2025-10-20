import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

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
X_train, X_test, y_train, y_test = train_test_split(
    data_X, data_Y, test_size=0.01, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.01, random_state=42)

# Netzwerkarchitektur definieren
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

# Training
# Parameter festlegen
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), weight_decay=1e-4)
batch_size = 16

# Dataset laden
Dataset=TensorDataset(X_train, y_train)
trainloader = DataLoader(Dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(X_val, y_val)
valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_dataset = TensorDataset(X_test, y_test)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


num_epochs=20

# Training starten
for epoch in range(num_epochs):
    correct = 0
    total=0
    train_loss= 0.0
    for i, data in enumerate(trainloader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total
    # ---------- Validation ----------
    net.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in valloader:
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total

    print(f"Epoch {epoch+1}/{num_epochs} "
          f"| Train Loss: {train_loss/len(trainloader):.4f} "
          f"| Train Acc: {train_accuracy:.2f}% "
          f"| Val Loss: {val_loss/len(valloader):.4f} "
          f"| Val Acc: {val_accuracy:.2f}%")

# ---------- Evaluation auf Testdaten ----------
net.eval()
test_correct = 0
test_total = 0
test_loss = 0.0

with torch.no_grad():
    for inputs, labels in testloader:
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_accuracy = 100 * test_correct / test_total
print(f"✅ Test Loss: {test_loss/len(testloader):.4f} | Test Accuracy: {test_accuracy:.2f}%")

torch.save(net, '../models/solar.pth')