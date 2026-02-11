class HyperConfiguration:
    def __init__(self, hidden_layer_size: int, learning_rate: float, momentum_coefficient: float):
        self.hidden_layer_size = hidden_layer_size
        self.learning_rate = learning_rate
        self.momentum_coefficient = momentum_coefficient