import torch
import torch.nn as nn

class LSTMModel(nn.Module):

    def __init__(self, input_size=4):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size,
            64,
            batch_first=True
        )

        self.fc = nn.Linear(64, 1)

    def forward(self, x):

        output, (hidden, cell) = self.lstm(x)

        return self.fc(hidden[-1])


model = LSTMModel()

print(model)