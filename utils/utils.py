import random
import torch
import numpy as np
from datetime import datetime
from typing import Any, Callable


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def benchmark(fun: Callable[[], Any]) -> tuple[Any, float]:
    start_time = datetime.now()
    ret = fun()
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    return ret, duration
