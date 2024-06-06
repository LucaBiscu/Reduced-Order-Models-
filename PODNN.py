import numpy as np
import torch
import torch.nn.functional as F

def projector(snapshot_matrix, basis, inner_product):
  bx = basis.T @ inner_product
  return np.linalg.solve(bx @ basis, bx @ snapshot_matrix.T).T

def train_podnn(network, steps, mus, projections, log=0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    network = network.to(device)
    optim = torch.optim.AdamW(network.parameters(), lr=1e-3, weight_decay=5e-2)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optim, 1000, gamma=0.95
    )
    mus, projections = torch.tensor(mus).float().to(device), torch.tensor(projections).float().to(device)
    for s in range(steps):
        optim.zero_grad()
        out = network(mus)
        loss = F.mse_loss(out, projections)
        loss.backward()
        optim.step()
        lr_scheduler.step()
        if log and (s % log == 0):
            print(
                f"Step {s}, lr {optim.param_groups[0]['lr']} loss {loss.item()}"
            )
    return network
