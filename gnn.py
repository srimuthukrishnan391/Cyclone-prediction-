import torch
import torch.nn as nn

class DynamicGNN(nn.Module):

    def __init__(self, input_size=4):
        super().__init__()

        self.layer = nn.Linear(
            input_size,
            128
        )

        self.output = nn.Linear(
            128,
            1
        )

    def forward(self, x):

        x = x[:, -1, :]

        x = torch.tanh(
            self.layer(x)
        )

        return self.output(x)


model = DynamicGNN()

print(model)