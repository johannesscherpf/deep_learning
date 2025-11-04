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
        self.inputsize = 1
        self.hiddensize = 10
        self.numlayers = 5
        self.outputsize = 7
        self.lstm = nn.LSTM(inputsize, hiddensize, numlayers, batch_first=True)
        self.fc = nn.Linear(hiddensize, outputsize)

    def forward(self, x):
        device = torch.device("cpu")
        h0 = torch.zeros(x.size(0), self.hiddensize).to(device)
        c0 = torch.zeros(x.size(0), self.hiddensize).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class Model:
    def __init__(self):
        # Gesamtes Modell laden
        script_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(script_dir, 'rnn.pth')

        self.LSTM = LSTM(inputsize=1, hiddensize=10, numlayers=5, outputsize=7)
        self.LSTM.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.LSTM.eval()

    def predict(self, test_x: np.ndarray) -> np.ndarray:
        # In Tensor konvertieren
        inputs = torch.from_numpy(test_x.astype(np.float32))
        inputs = inputs.view(-1, 90, 1)

        with torch.no_grad():
            outputs = self.LSTM(inputs)
            predicted = outputs.numpy()
        return predicted