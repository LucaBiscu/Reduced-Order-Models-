import torch.nn as nn
from PiNN import *

network = MLPiNN(4, 1, 10, 4, activation=nn.Mish)
train_pinn(
    network,
    4000,
    100,
    torch.tensor([0.0, 1.0]),
    torch.tensor([0.1, 1.0]),
    log=100,
)
