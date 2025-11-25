import torch


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
            nn.MaxPool2d(2, 2),  # Output: 16x16

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 8x8

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 4x4
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
        """
        Initialize the Model with the target categories.
        """
        self.categories = ["airplane", "automobile", "bird", "cat", "deer", "dog", "horse", "ship", "truck"]

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        x_test = torch.from_numpy(x_test)
        transform_val = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.4936, 0.4869, 0.4577), (0.2030, 0.2018, 0.2049))])

        test_dataset = Dataset(x_test, transforms=transform_val)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

