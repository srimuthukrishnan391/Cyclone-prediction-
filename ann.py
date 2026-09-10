import torch
import torch.nn as nn

class ANN(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)


# Example
model = ANN(4)

print(model)