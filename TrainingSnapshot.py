from typing import Any

class TrainingSnapshot:
    def __init__(self, state_dict: dict[str, Any], hidden_layer_size: int = 0, lr: float = 0.0, momentum: float = 0.0, epoch: int = 0, val_loss: float = float('inf')):
        self.state_dict = state_dict
        self.hidden_layer_size = hidden_layer_size
        self.lr = lr
        self.momentum = momentum
        self.epoch = epoch
        self.val_loss = val_loss