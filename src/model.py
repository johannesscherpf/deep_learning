import torch
import numpy as np
import os

class Model:
    def __init__(self):
        # Gesamtes Modell laden
        model_path = os.path.join(os.path.dirname(__file__), "solar.pth")
        self.net = torch.load(model_path, map_location="cpu")
        #self.net = torch.load("solar.pth", map_location="cpu")
        self.net.eval()

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        # In Tensor konvertieren
        inputs = (x_test - x_test.mean) / x_test.std
        inputs = torch.from_numpy(inputs.astype(np.float32))

        with torch.no_grad():
            outputs = self.net(inputs)
            _, predicted = torch.max(outputs, dim=1)
        return predicted.numpy()