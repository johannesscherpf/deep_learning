import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import numpy as np
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split


# Check if CUDA (GPU support) is available
is_cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {is_cuda_available}")
print(torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = pd.read_csv('../data/rnn_train.csv')

def create_sequences(data, stride, window_size, target_size):
    sequences = []
    targets=[]
    for number in range((len(data)-window_size - target_size)//stride):
        start=number*stride
        end=start + window_size
        sequence = data[start:end]
        target = data[end:end + target_size]
        targets.append(target)
        sequences.append(sequence)
    return sequences, targets

def sequences_for_all_decades(data, stride, window_size, target_size):
    sequences = []
    targets=[]
    for row in range(len(data)):
        data_copy = data.copy()
        data_row=data_copy.iloc[row].tolist()
        x_values, y_values = create_sequences(data_row, stride, window_size, target_size)
        sequences.extend(x_values)
        targets.extend(y_values)
    return np.array(sequences), np.array(targets)

def split(sequences, targets, test_size=0.2):
    test_size=int(test_size*len(sequences))
    sequences = np.array(sequences)
    targets = np.array(targets)
    trainsequences, testsequences = sequences[:-test_size], sequences[-test_size:]
    traintargets, testtargets = targets[:-test_size],  targets[-test_size:]
    return trainsequences, testsequences, traintargets, testtargets

sequences, targets = sequences_for_all_decades(data, 1, 90, 7)
trainsequences, testsequences, traintargets, testtargets = split(sequences, targets, test_size=0.2)
dataset = TensorDataset(Tensor(trainsequences), Tensor(traintargets))
val_dataset = TensorDataset(Tensor(testsequences), Tensor(testtargets))
valdataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)
train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
inputs=train_dataloader

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(5, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

class LSTM(nn.Module):
    def __init__(self, inputsize, hiddensize, numlayers, outputsize):
        super(LSTM, self).__init__()
        self.inputsize = inputsize
        self.hiddensize = hiddensize
        self.numlayers = numlayers
        self.outputsize = outputsize
        self.lstm = nn.LSTM(inputsize, hiddensize, numlayers, batch_first=True)
        self.fc = nn.Linear(hiddensize, outputsize)

    def forward(self, x):
        h0 = torch.zeros(self.numlayers, x.size(0), self.hiddensize).to(x.device)
        c0 = torch.zeros(self.numlayers, x.size(0), self.hiddensize).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Hyperparameters
input_size = 1
hidden_size = 10
num_layers = 5
output_size = 7
batch_size = 256
seq_len = 90


# Create the model
model = RNN(input_size, hidden_size, num_layers, output_size)
#model = LSTM(input_size, hidden_size, num_layers, output_size)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100

"""
for epoch in range(num_epochs):
    for batch in train_dataloader:
        inputs, targets = batch
        inputs = inputs.view(-1, seq_len, 1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
"""

class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0

    def call(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

early_stopping = EarlyStopping(10)


def calculatevalloss(model, val_dataloader):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            inputs, targets = batch
            inputs = inputs.view(-1, seq_len, 1)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    val_loss /= len(valdataloader)
    return val_loss

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        inputs, targets = batch
        inputs = inputs.view(-1, seq_len, 1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    total_loss /= len(train_dataloader)
    model.eval()
    val_loss = calculatevalloss(model, valdataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}')
    if early_stopping.call(val_loss):
        break

#torch.save(model.state_dict(), '../models/rnn.pth')