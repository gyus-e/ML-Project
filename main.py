# Giuseppe Amato DE5000051   -   Giuseppe Di Martino DE5000042
# P11)
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
import logging
import random
import numpy as np
from datetime import datetime

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from MyNeuralNetwork import MyNeuralNetwork
from utils import train_loop, test_loop

DATA_DIR = "data"
LOGS_DIR = "logs"

IMG_SIZE = 28 * 28
NUM_CLASSES = 10

BATCH_SIZE = 1024
NUM_WORKERS = 4

EPOCHS = 5
RANDOM_SEEDS = [51, 2167]
HIDDEN_LAYER_SIZES = [64, 128, 256, 512, 1024]
LEARNING_RATES = [0.01, 0.1, 0.5]
MOMENTUM_COEFFICIENTS = [0.1, 0.5, 0.9]


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H-%M-%S")
    logging.basicConfig(
        filename=f"{LOGS_DIR}/{timestamp}.csv", level=logging.INFO, format="%(message)s"
    )
    logging.info(
        "device;training_samples;validation_samples;test_samples;batch_size;loss_function;epochs;random_seed;hidden_layer_size;learning_rate;momentum;epoch;accuracy;avg_loss;epoch_duration_seconds"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_training_data = MNIST(
        root=DATA_DIR, train=True, download=True, transform=ToTensor()
    )

    test_data = MNIST(root=DATA_DIR, train=False, download=True, transform=ToTensor())

    train_size = int(0.8 * len(full_training_data))
    val_size = len(full_training_data) - train_size

    test_dataloader: DataLoader[MNIST] = DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    loss_fn = nn.CrossEntropyLoss()

    for seed in RANDOM_SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        training_data, validation_data = random_split(
            full_training_data, [train_size, val_size]
        )

        train_dataloader: DataLoader[MNIST] = DataLoader(
            training_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
        )
        validation_dataloader: DataLoader[MNIST] = DataLoader(
            validation_data,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )

        for hidden_layer_size in HIDDEN_LAYER_SIZES:
            for lr in LEARNING_RATES:
                for momentum in MOMENTUM_COEFFICIENTS:

                    model = MyNeuralNetwork(
                        input_layer_size=IMG_SIZE,
                        hidden_layer_size=hidden_layer_size,
                        output_layer_size=NUM_CLASSES,
                    ).to(device)

                    # Stochastic Gradient Descent
                    optimizer = optim.SGD(
                        model.parameters(),
                        lr=lr,
                        momentum=momentum,
                        weight_decay=0.0,
                    )

                    for epoch in range(EPOCHS):
                        start_time = datetime.now()

                        train_loop(train_dataloader, model, loss_fn, optimizer)
                        val_loss, correct = test_loop(
                            validation_dataloader, model, loss_fn
                        )

                        end_time = datetime.now()
                        epoch_duration = (end_time - start_time).total_seconds()
                        logging.info(
                            f"{device};{train_size};{val_size};{len(test_data)};{BATCH_SIZE};{loss_fn};{EPOCHS};{seed};{hidden_layer_size};{lr};{momentum};{epoch+1};{(100*correct):>0.1f}%;{val_loss:>8f};{epoch_duration:>8f}"
                        )

                    start_time = datetime.now()

                    test_loss, test_correct = test_loop(test_dataloader, model, loss_fn)

                    end_time = datetime.now()
                    epoch_duration = (end_time - start_time).total_seconds()
                    logging.info(
                        f"{device};{train_size};{val_size};{len(test_data)};{BATCH_SIZE};{loss_fn};{EPOCHS};{seed};{hidden_layer_size};{lr};{momentum};TEST;{(100*test_correct):>0.1f}%;{test_loss:>8f};{epoch_duration:>8f}"
                    )


if __name__ == "__main__":
    main()
