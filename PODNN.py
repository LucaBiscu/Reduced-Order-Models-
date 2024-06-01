

import numpy as np
import GeDiM4Py as gedim
from functools import partial
from weakforms import *
from FOM import newton_solver
import torch
import torch.nn as nn
import torch.nn.functional as F



def snapshotProjector(snapshot_matrix, basis, inner_product):
  #inner_product, _ = gedim.AssembleStiffnessMatrix(ones, problem_data, lib) #valutare se farlo passare come output da pod_basis
  reduced_inner_product = basis.T @ inner_product @ basis
  #x_train = torch.tensor(np.float32(training_set))
  y_train = []
  for i in range(snapshot_matrix.shape[0]):
    snapshot_to_project = snapshot_matrix[i]
    projected_snapshot = np.linalg.solve(reduced_inner_product, np.transpose(basis)@inner_product@snapshot_to_project)
    y_train.append(projected_snapshot)
  return torch.tensor(y_train, dtype=torch.float32)



class podNN(nn.Module):
    def __init__(
        self, out_dim, in_dim, hidden_size, hidden_layers, activation=nn.Tanh
    ):
        super(podNN, self).__init__()
        assert hidden_layers > 0, "Number of hidden layers must be positive"
        assert hidden_size > 0, "Dimension of hidden layer must be non zero"
        layers = [nn.Linear(in_dim, hidden_size), activation()]
        for layer in range(hidden_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation())
        layers.append(nn.Linear(hidden_size, out_dim))
        self.layers = nn.Sequential(*layers)
        self.loss = nn.MSELoss()

    def forward(self, mu):
        return self.layers(mu)



