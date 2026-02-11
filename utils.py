import logging
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

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

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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

            pred = model(X)
            loss = loss_fn(pred, y)

            test_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    return test_loss, correct

