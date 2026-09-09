import numpy as np
from dataclasses import dataclass
from typing import Optional

def frange(start:float,stop:float,step:float) -> np.ndarray:
    """Generate a range of floating-point numbers from start to stop with a given step size."""
    numbers = []
    current = start
    while current < stop:
        numbers.append(current)
        current += step
    return np.array(numbers)

def irange(start:int,stop:int,step:int=1) -> np.ndarray:
    """Generate a range of integers from start to stop with a given step size."""
    numbers = []
    for num in range(start, stop, step):
        numbers.append(num)
    return np.array(numbers)

@dataclass
class Hyperparameters:
    num_layer: np.ndarray
    dropout: np.ndarray
    sequence_length: np.ndarray
    batch_size: np.ndarray
    epochs: np.ndarray
    learning_rate: np.ndarray
    
@dataclass
class Parameters:
    #data level
    sequence_length: int
    #model level
    num_layer: int
    dropout: float
    #training level
    epochs: int
    batch_size: int
    learning_rate: float

default_hyperparameters = Hyperparameters(
    num_layer=[30],          # 10 to 100 in steps of 10
    dropout=[0.25],         # 0.1 to 0.5 in steps of 0.1
    sequence_length=[100],   # 50 to 200 in steps of 25
    batch_size=[128],        # 32 to 256 in steps of 32
    epochs=[20],               # 5 to 50 in steps of 5
    learning_rate=[0.0001]  # 0.0001 to 0.001 in steps of 0.0001
)

coarse_hyperparameters = Hyperparameters(
    num_layer=np.array([10, 50]),
    dropout=np.array([0.1, 0.5]),
    sequence_length=np.array([50, 200]),
    batch_size=np.array([32, 256]),
    epochs=np.array([5, 20]),
    learning_rate=np.array([0.0001, 0.001])
)

fine_hyperparameters = Hyperparameters(
    num_layer=np.array([30, 40, 50]),
    dropout=np.array([0.3]),
    sequence_length=np.array([150, 175, 200]),
    batch_size=np.array([64]),
    epochs=np.array([10]),
    learning_rate=np.array([0.0005, 0.00075, 0.001])
)

fine_hyperparameters2 = Hyperparameters(
    num_layer=np.array([30]),
    dropout=np.array([0.3]),
    sequence_length=np.array([100,115,130]),
    batch_size=np.array([64]),
    epochs=np.array([10]),
    learning_rate=np.array([0.0005])
)

postprocess_hyperparameters = {
    'k': [2, 3, 4, 5],
    'tau': [0.4, 0.5, 0.6],
    'tau_trans': [0.5, 0.65, 0.8],
    'force_tau': [0.8, 0.9, 1.0]
}