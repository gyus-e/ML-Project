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
import itertools
import numpy as np
from datetime import datetime

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from MyNeuralNetwork import MyNeuralNetwork
from utils import train_loop, test_loop, benchmark

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
        "device;training_samples;validation_samples;test_samples;batch_size;loss_function;random_seed;hidden_layer_size;learning_rate;momentum;epochs;epoch;phase;accuracy;loss;duration"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # transform viene applicato a ogni immagine quando viene caricata.
    # Ci limitiamo a trasformarle in matrici 28x28 (dimensione in pixel delle immagini di MNIST), senza normalizzarle.
    full_training_data = MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data = MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=ToTensor(),
    )

    train_size = int(0.8 * len(full_training_data))
    val_size = len(full_training_data) - train_size
    test_size = len(test_data)

    test_dataloader: DataLoader[MNIST] = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
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
            training_data,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
        )
        validation_dataloader: DataLoader[MNIST] = DataLoader(
            validation_data,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )

        for hidden_layer_size, lr, momentum in itertools.product(
            HIDDEN_LAYER_SIZES, LEARNING_RATES, MOMENTUM_COEFFICIENTS
        ):
            # È importante ricreare il modello ogni volta che cambia un iperparametro, non solo quando cambia il layer size, 
            # perché altrimenti verrebbero riutilizzati i pesi del training precedente
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
                (train_loss, train_correct), train_time = benchmark(
                    lambda model=model, optimizer=optimizer: train_loop(
                        train_dataloader, model, loss_fn, optimizer
                    )
                )
                logging.info(
                    f"{device};{train_size};{val_size};{test_size};{BATCH_SIZE};{loss_fn};{seed};{hidden_layer_size};{lr};{momentum};{EPOCHS};{epoch+1};TRAIN;{(100*train_correct):>0.1f}%;{train_loss:>8f};{train_time:>8f}"
                )

                (val_loss, val_correct), val_time = benchmark(
                    lambda model=model: test_loop(
                        validation_dataloader, model, loss_fn
                    )
                )
                logging.info(
                    f"{device};{train_size};{val_size};{test_size};{BATCH_SIZE};{loss_fn};{seed};{hidden_layer_size};{lr};{momentum};{EPOCHS};{epoch+1};VAL;{(100*val_correct):>0.1f}%;{val_loss:>8f};{val_time:>8f}"
                )

            (test_loss, test_correct), test_time = benchmark(
                lambda model=model: test_loop(
                    test_dataloader, model, loss_fn
                )
            )
            logging.info(
                f"{device};{train_size};{val_size};{test_size};{BATCH_SIZE};{loss_fn};{seed};{hidden_layer_size};{lr};{momentum};{EPOCHS};;TEST;{(100*test_correct):>0.1f}%;{test_loss:>8f};{test_time:>8f}"
            )


if __name__ == "__main__":
    main()
