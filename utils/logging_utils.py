import logging
import torch
from torch import nn

from utils.HyperConfiguration import HyperConfiguration
from utils.StepData import StepData


def log_csv_head():
    logging.info(
        "device;training_samples;validation_samples;test_samples;batch_size;loss_function;random_seed;hidden_layer_size;learning_rate;momentum;epochs;epoch;phase;accuracy;loss;duration"
    )


def log_data(
    device: torch.device,
    data_sizes: tuple[int, int, int],
    batch_size: int,
    loss_fn: nn.Module,
    seed: int,
    epochs: int,
    current_hyper_config: HyperConfiguration,
    model_data: tuple[list[tuple[StepData, StepData]], StepData],
):
    train_size, val_size, test_size = data_sizes
    all_epochs_data, test_data = model_data
    for epoch, (train_data, val_data) in enumerate(all_epochs_data):
        logging.info(
            f"{device};{train_size};{val_size};{test_size};{batch_size};{loss_fn};{seed};{current_hyper_config.hidden_layer_size};{current_hyper_config.learning_rate};{current_hyper_config.momentum_coefficient};{epochs};{epoch+1};TRAIN;{(100*train_data.correct):>0.1f}%;{train_data.loss:>8f};{train_data.duration:>8f}"
        )
        logging.info(
            f"{device};{train_size};{val_size};{test_size};{batch_size};{loss_fn};{seed};{current_hyper_config.hidden_layer_size};{current_hyper_config.learning_rate};{current_hyper_config.momentum_coefficient};{epochs};{epoch+1};VAL;{(100*val_data.correct):>0.1f}%;{val_data.loss:>8f};{val_data.duration:>8f}"
        )
    logging.info(
        f"{device};{train_size};{val_size};{test_size};{batch_size};{loss_fn};{seed};{current_hyper_config.hidden_layer_size};{current_hyper_config.learning_rate};{current_hyper_config.momentum_coefficient};{epochs};;TEST;{(100*test_data.correct):>0.1f}%;{test_data.loss:>8f};{test_data.duration:>8f}"
    )
