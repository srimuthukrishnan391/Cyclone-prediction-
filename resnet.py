import torch
import torch.nn as nn

class ResNetSimple(nn.Module):

    def __init__(self, input_size=4):
        super().__init__()

        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, x):

        x = x[:, -1, :]

        identity = self.layer1(x)

        out = torch.relu(identity)
        out = torch.relu(self.layer2(out))

        # Residual connection
        out = out + identity

        return self.output(out)


model = ResNetSimple()

print(model)