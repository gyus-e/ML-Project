import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

DATA_DIR = "data"

FULL_TRAINING_DATA = MNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=ToTensor(),
)

TEST_DATA = MNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=ToTensor(),
)

x = torch.cat([FULL_TRAINING_DATA[i][0] for i in range(len(FULL_TRAINING_DATA))], dim=0)
img = FULL_TRAINING_DATA[0][0]

print(
    f"""size: {img.size()}
      mean: {x.mean().item()}
      std: {x.std().item()}"""
)
