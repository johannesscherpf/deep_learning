import torch
import numpy as np

class Model:
    def __init__(self):
        # Gesamtes Modell laden
        self.net = torch.load("solar.pth", map_location="cpu")
        self.net.eval()

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        # In Tensor konvertieren
        inputs = torch.from_numpy(x_test.astype(np.float32))
        with torch.no_grad():
            outputs = self.net(inputs)
            _, predicted = torch.max(outputs, dim=1)
        return predicted.numpy()