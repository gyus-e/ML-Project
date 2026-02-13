from torch import nn

class MyNeuralNetwork(nn.Module):
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size):
        super().__init__()
        
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.input_layer_size, self.hidden_layer_size),
            nn.ReLU(), # Hidden layer activation
            nn.Linear(self.hidden_layer_size, self.output_layer_size),
            # Output activation: identity (logits), perché CrossEntropyLoss include softmax
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
