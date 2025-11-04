import torch
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

# --- gleiche Architektur wie beim Training ---
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
batch_size = 64
seq_len = 90

class Model:
    def __init__(self):
        # Gesamtes Modell laden
        script_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(script_dir, 'rnn.pth')

        self.LSTM = LSTM()
        self.LSTM.load_state_dict(torch.load(model_path, map_location="cpu"))
        #self.net = torch.load(model_path, map_location="cpu", weights_only=False)
        #self.net = torch.load("solar.pth", map_location="cpu")
        self.LSTM.eval()

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        # In Tensor konvertieren
        inputs = torch.fromnumpy(test_x.astype(np.float32))


        with torch.no_grad():
            outputs = self.LSTM(inputs)
            predicted = outputs.numpy()
        return predicted.numpy()