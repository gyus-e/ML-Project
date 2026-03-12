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
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from sklearn.model_selection import train_test_split

from MyNeuralNetwork import MyNeuralNetwork
from TrainingSnapshot import TrainingSnapshot
from utils import train_loop, test_loop, benchmark

DATA_DIR = "data"
LOGS_DIR = "logs"
MODELS_DIR = "models"

IMG_SIZE = 28 * 28
NUM_CLASSES = 10
MNIST_MEAN = 0.13066047430038452
MNIST_STD = 0.30810782313346863

BATCH_SIZE = 100
NUM_WORKERS = 4

EPOCHS = 30
RANDOM_SEEDS = [43, 689, 5093]
LEARNING_RATES = [1e-3, 1e-2, 1e-1]
MOMENTUM_COEFFICIENTS = [0.1, 0.5, 0.9]
HIDDEN_LAYER_SIZES = [
    int(np.sqrt(IMG_SIZE)),
    int(np.average([IMG_SIZE, NUM_CLASSES])),
    IMG_SIZE,
    IMG_SIZE * 2,
    IMG_SIZE * 10,
]

# transform viene applicato a ogni immagine quando viene caricata.
# ToTensor le trasforma in matrici 28x28 (dimensione in pixel delle immagini di MNIST) e applica Min-Max Normalization (scala i valori da 0-255 a 0-1).
# Normalize standardizza i valori sottraendo la media e dividendo per la deviazione standard del dataset MNIST.
TRANSFORM = Compose(
    [
        ToTensor(),
        Normalize(mean=MNIST_MEAN, std=MNIST_STD),
    ]
)

FULL_TRAINING_DATA = MNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=TRANSFORM,
)

TEST_DATA = MNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=TRANSFORM,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOSS_FN = nn.CrossEntropyLoss()

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H-%M-%S")
    logging.basicConfig(
        filename=f"{LOGS_DIR}/{timestamp}.csv", level=logging.INFO, format="%(message)s"
    )
    logging.info(
        "device;data_samples;batch_size;loss_function;random_seed;hidden_layer_size;hidden_layer_activation;learning_rate;momentum;epochs;epoch;phase;accuracy;loss;duration"
    )
    current_models_dir = os.path.join(MODELS_DIR, timestamp)
    os.makedirs(current_models_dir, exist_ok=True)

    for seed in RANDOM_SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        train_idx, val_idx = train_test_split(
            range(len(FULL_TRAINING_DATA)),
            test_size=0.2,
            train_size=0.8,
            random_state=seed,
            shuffle=True,
            stratify=FULL_TRAINING_DATA.targets.numpy(),
        )

        training_data = Subset(FULL_TRAINING_DATA, train_idx)
        validation_data = Subset(FULL_TRAINING_DATA, val_idx)

        train_size = len(training_data)  # 48,000 samples for training
        val_size = len(validation_data)  # 12,000 samples for validation
        test_size = len(TEST_DATA)  # 10,000 samples for testing

        train_dataloader: DataLoader[MNIST] = DataLoader(
            training_data,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            persistent_workers=True,
            pin_memory=True,
        )

        validation_dataloader: DataLoader[MNIST] = DataLoader(
            validation_data,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            persistent_workers=True,
            pin_memory=True,
        )

        best_model: TrainingSnapshot | None = None

        for hidden_layer_size, lr, momentum in itertools.product(
            HIDDEN_LAYER_SIZES, LEARNING_RATES, MOMENTUM_COEFFICIENTS
        ):
            # È importante ricreare il modello ogni volta che cambia un iperparametro, non solo quando cambia il layer size,
            # perché altrimenti verrebbero riutilizzati i pesi del training precedente
            model = MyNeuralNetwork(
                input_layer_size=IMG_SIZE,
                hidden_layer_size=hidden_layer_size,
                output_layer_size=NUM_CLASSES,
            ).to(DEVICE, non_blocking=True)

            # Stochastic Gradient Descent
            optimizer = optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=0.0,
            )

            hl_activation = model.feedforward_network[1]

            for epoch in range(EPOCHS):
                (train_loss, train_correct), train_time = benchmark(
                    lambda model=model, optimizer=optimizer: train_loop(
                        train_dataloader, model, LOSS_FN, optimizer
                    )
                )
                logging.info(
                    f"{DEVICE};{train_size};{BATCH_SIZE};{LOSS_FN};{seed};{hidden_layer_size};{hl_activation};{lr};{momentum};{EPOCHS};{epoch+1};TRAIN;{(100*train_correct):>0.1f};{train_loss:>8f};{train_time:>8f}"
                )

                (val_loss, val_correct), val_time = benchmark(
                    lambda model=model: test_loop(validation_dataloader, model, LOSS_FN)
                )
                logging.info(
                    f"{DEVICE};{val_size};{BATCH_SIZE};{LOSS_FN};{seed};{hidden_layer_size};{hl_activation};{lr};{momentum};{EPOCHS};{epoch+1};VAL;{(100*val_correct):>0.1f};{val_loss:>8f};{val_time:>8f}"
                )

                if best_model is None or val_loss < best_model.val_loss:
                    best_model = TrainingSnapshot(
                        model.state_dict(),
                        hidden_layer_size,
                        lr,
                        momentum,
                        epoch,
                        val_loss,
                    )

        if best_model is not None:
            torch.save(
                best_model.state_dict,
                os.path.join(
                    current_models_dir,
                    f"model_seed{seed}_size{best_model.hidden_layer_size}_lr{best_model.lr}_mom{best_model.momentum}_epoch{best_model.epoch}.pth",
                ),
            )

            model = MyNeuralNetwork(
                input_layer_size=IMG_SIZE,
                hidden_layer_size=best_model.hidden_layer_size,
                output_layer_size=NUM_CLASSES,
            ).to(DEVICE, non_blocking=True)

            model.load_state_dict(best_model.state_dict)

            hl_activation = model.feedforward_network[1]

            test_dataloader: DataLoader[MNIST] = DataLoader(
                TEST_DATA,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                persistent_workers=True,
                pin_memory=True,
            )

            (test_loss, test_correct), test_time = benchmark(
                lambda model=model, test_dataloader=test_dataloader: test_loop(
                    test_dataloader, model, LOSS_FN
                )
            )
            logging.info(
                f"{DEVICE};{test_size};{BATCH_SIZE};{LOSS_FN};{seed};{best_model.hidden_layer_size};{hl_activation};{best_model.lr};{best_model.momentum};{EPOCHS};;TEST;{(100*test_correct):>0.1f};{test_loss:>8f};{test_time:>8f}"
            )


if __name__ == "__main__":
    main()
