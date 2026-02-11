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
import itertools
from datetime import datetime
from typing import Callable, List

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from MyNeuralNetwork import MyNeuralNetwork
from utils.StepData import StepData
from utils.HyperConfiguration import HyperConfiguration
from utils.utils import set_seed, benchmark
from utils.logging_utils import log_csv_head, log_data

DATA_DIR = "data"
LOGS_DIR = "logs"
MODELS_DIR = "model"

IMG_SIZE = 28 * 28
NUM_CLASSES = 10

BATCH_SIZE = 1024
NUM_WORKERS = 4

EPOCHS = 5
RANDOM_SEEDS = [17, 132]
HIDDEN_LAYER_SIZES = [64, 128, 256, 512, 1024]
LEARNING_RATES = [0.01, 0.1, 0.5]
MOMENTUM_COEFFICIENTS = [0.1, 0.5, 0.9]


def main():
    log_csv_head()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    sizes = (train_size, val_size, test_size)

    test_dataloader: DataLoader[MNIST] = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    loss_fn = nn.CrossEntropyLoss()

    for seed in RANDOM_SEEDS:
        set_seed(seed)

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

        configurations = itertools.product(
            HIDDEN_LAYER_SIZES, LEARNING_RATES, MOMENTUM_COEFFICIENTS
        )

        best_loss = float("inf")
        best_model = None

        for hidden_layer_size, lr, momentum in configurations:
            hyper_conf = HyperConfiguration(hidden_layer_size, lr, momentum)

            model = MyNeuralNetwork(
                input_layer_size=IMG_SIZE,
                hidden_layer_size=hyper_conf.hidden_layer_size,
                output_layer_size=NUM_CLASSES,
            ).to(device)

            # Stochastic Gradient Descent
            optimizer = optim.SGD(
                model.parameters(),
                lr=hyper_conf.learning_rate,
                momentum=hyper_conf.momentum_coefficient,
                weight_decay=0.0,
            )

            all_epochs_data: List[tuple[StepData, StepData]] = []

            for _ in range(EPOCHS):
                (train_loss, train_correct), train_duration = benchmark(
                    lambda model=model, optimizer=optimizer: train_loop(
                        train_dataloader, model, loss_fn, optimizer
                    )
                )
                (val_loss, val_correct), val_duration = benchmark(
                    lambda model=model: test_loop(validation_dataloader, model, loss_fn)
                )

                train_data = StepData(train_loss, train_correct, train_duration)
                val_data = StepData(val_loss, val_correct, val_duration)
                epoch_data = (train_data, val_data)
                all_epochs_data.append(epoch_data)

            (test_loss, test_correct), test_duration = benchmark(
                lambda model=model: test_loop(test_dataloader, model, loss_fn)
            )
            test_data = StepData(test_loss, test_correct, test_duration)
            model_data = (all_epochs_data, test_data)

            if test_loss < best_loss:
                best_loss = test_loss
                best_model = model

            log_data(
                device, sizes, BATCH_SIZE, loss_fn, seed, EPOCHS, hyper_conf, model_data
            )
        
        if best_model is not None:
            torch.save(best_model.state_dict(), f"{MODELS_DIR}/f{seed}.pth")


def train_loop(dataloader: DataLoader[Dataset], model: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer) -> tuple[float, float]:
    device = model.parameters().__next__().device
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0
    correct = 0

    model.train()
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)

        pred, loss = predict(X, y, model, loss_fn, lambda loss: evaluate(optimizer, loss))

        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    train_loss /= num_batches
    correct /= size
    
    return train_loss, correct


def test_loop(dataloader: DataLoader[Dataset], model: nn.Module, loss_fn: nn.Module) -> tuple[float, float]:
    device = model.parameters().__next__().device
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    correct = 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            pred, loss = predict(X, y, model, loss_fn, lambda loss: None)

            test_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    return test_loss, correct


def predict(X, y, model: nn.Module, loss_fn: nn.Module, evaluate_func: Callable[..., None]):
    pred = model(X)
    loss = loss_fn(pred, y)
    evaluate_func(loss)
    return pred, loss


def evaluate(optimizer: torch.optim.Optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H-%M-%S")
    logging.basicConfig(
        filename=f"{LOGS_DIR}/{timestamp}.csv", level=logging.INFO, format="%(message)s"
    )

    main()
