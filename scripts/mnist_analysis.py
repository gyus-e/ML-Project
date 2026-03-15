import torch
from torch.utils.data import Subset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split


RANDOM_SEEDS = [43, 689, 5093]

DATA_DIR = "data"

FULL_TRAINING_DATA = MNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=ToTensor(),  # applica giá min-max normalization
)

TEST_DATA = MNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=ToTensor(),
)

for seed in RANDOM_SEEDS:
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

    images = torch.cat([X for (X, y) in training_data], dim=0)
    val_images = torch.cat([X for (X, y) in validation_data], dim=0)

    print(
        f"""train mean: {images.mean().item()}
        train std: {images.std().item()}
        val mean: {val_images.mean().item()}
        val std: {val_images.std().item()}"""
    )
