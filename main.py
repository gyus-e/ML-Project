# P11   -   Giuseppe Amato DE5000051   -   Giuseppe Di Martino DE5000042

# Use the raw MNIST images as input for a 10-class classification task.
# Build a dataset of N input–label pairs and split it into training and test sets
# (at least 10,000 training samples and 2,500 test samples).
# Train the models using gradient descent with momentum.
# Study the learning behavior of a neural network with a single hidden layer
# by varying the learning rate η, the momentum coefficient, and the number of hidden units
# (use at least five different hidden-layer sizes).
# Keep all other architectural choices fixed, including the output activation functions.
# For each configuration, analyze the training process
# (epochs to convergence, trends in training and validation errors, test accuracy).
# Additionally, compare the stability of training under different hyperparameter settings, examine the sensitivity to initialization
# (run each experiment at least twice with different random seeds),
# and report any systematic patterns you observe.

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from MyNeuralNetwork import MyNeuralNetwork


HIDDEN_LAYER_SIZES = [128, 256, 512, 1024, 2048]
MOMENTUM_COEFFICIENTS = [0.1, 0.5, 0.9]
LEARNING_RATES = [1e-3, 1e-2, 1e-1]
ROOT_DIR = "data"


training_data = datasets.MNIST(
    root=ROOT_DIR, train=True, download=True, transform=ToTensor()
)

test_data = datasets.MNIST(
    root=ROOT_DIR, train=False, download=True, transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True, num_workers=0)

test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=0)

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

model = MyNeuralNetwork(hidden_layer_size=HIDDEN_LAYER_SIZES[0]).to(device)

print(model.hidden_layer_size)
