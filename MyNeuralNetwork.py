from torch import nn

INPUT_IMG_SIZE = 28*28
OUTPUT_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

class MyNeuralNetwork(nn.Module):
    def __init__(self, hidden_layer_size):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(INPUT_IMG_SIZE, self.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(self.hidden_layer_size, len(OUTPUT_CLASSES)),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
