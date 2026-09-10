import torch
import torch.nn as nn

class CNN(nn.Module):

    def __init__(self, features=4):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=6,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.fc = nn.Linear(
            16 * features,
            1
        )

    def forward(self, x):

        x = torch.relu(self.conv(x))

        x = x.reshape(x.size(0), -1)

        return self.fc(x)


model = CNN()

print(model)