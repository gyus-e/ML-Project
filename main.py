# Giuseppe Amato DE5000051   -   Giuseppe Di Martino DE5000042
# P11
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

import os
from datetime import datetime
import logging
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from MyNeuralNetwork import MyNeuralNetwork
from utils import train_loop, test_loop

DATA_DIR = "data"
LOGS_DIR = "logs"

BATCH_SIZE = 64
EPOCHS = 5
HIDDEN_LAYER_SIZES = [128, 256, 512, 1024, 2048]
MOMENTUM_COEFFICIENTS = [0.1, 0.5, 0.9]
LEARNING_RATES = [1e-3, 1e-2, 1e-1]


os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
logging.basicConfig(
    filename=f"{LOGS_DIR}/{timestamp}.log", level=logging.INFO, format="%(message)s"
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

training_data = MNIST(root=DATA_DIR, train=True, download=True, transform=ToTensor())
test_data = MNIST(root=DATA_DIR, train=False, download=True, transform=ToTensor())

train_dataloader: DataLoader[MNIST] = DataLoader(
    training_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
test_dataloader: DataLoader[MNIST] = DataLoader(
    test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)

loss_fn = nn.CrossEntropyLoss()


logging.info(
    f"Device: {device}\nTraining samples: {len(training_data)}\nTest samples: {len(test_data)}\nBatch size: {BATCH_SIZE}\nEpochs: {EPOCHS}\n-------------------------------"
)

for hidden_layer_size in HIDDEN_LAYER_SIZES:
    for momentum in MOMENTUM_COEFFICIENTS:
        for lr in LEARNING_RATES:
            logging.info(
                f"Hidden layer size: {hidden_layer_size}\nMomentum: {momentum}\nLearning rate: {lr}\n"
            )

            model = MyNeuralNetwork(hidden_layer_size=hidden_layer_size).to(device)

            # Stochastic Gradient Descent
            optimizer = optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=0.0,
            )

            for epoch in range(EPOCHS):
                logging.info(f"Epoch {epoch+1}")
                train_loop(train_dataloader, model, loss_fn, optimizer)
                test_loop(test_dataloader, model, loss_fn)
