import logging
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

def train_loop(dataloader: DataLoader[Dataset], model: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer):
    device = model.parameters().__next__().device
    size = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss = loss.item()
            current = batch * batch_size + len(X)
            logging.debug(f"\t\tloss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test_loop(dataloader: DataLoader[Dataset], model: nn.Module, loss_fn: nn.Module):
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
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size 
    logging.info(f"\t\tAccuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
