import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPiNN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_size, hidden_layers, activation=nn.Tanh):
        super(MLPiNN, self).__init__()
        assert hidden_layers > 0, "Number of hidden layers must be positive"
        assert hidden_size > 0, "Dimension of hidden layer must be non zero"
        layers = [nn.Linear(in_dim, hidden_size), activation()]
        for layer in range(hidden_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation())
        layers.append(nn.Linear(hidden_size, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, inp):
        return self.layers(inp)


def boundary_loss(net, x, mu):
    inp = torch.cat((x, mu), dim=1)
    return (net(inp) ** 2).mean()


def residual_loss(net, x, mu):
    assert x.shape[1] == 2, "x must be 2 dimensional"
    assert mu.shape[1] == 2, "mu must be 2 dimensional"
    assert torch.all(mu[:, 1] != 0), "mu 1 must not be zero"
    x.requires_grad = True
    inp = torch.cat((x, mu), dim=1)
    u = net(inp).squeeze()
    grad = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    laplacian = torch.autograd.grad(grad[:, 0], x, torch.ones_like(grad[:, 0]), create_graph=True, retain_graph=True)[0][:, 0]
    laplacian += torch.autograd.grad(grad[:, 1], x, torch.ones_like(grad[:, 1]), create_graph=True, retain_graph=True)[0][:, 1]
    x = x.detach()
    g = 100 * torch.sin(2 * torch.pi * x[:, 0]) * torch.cos(2 * torch.pi * x[:, 1])
    r = -laplacian + (mu[:, 0] / mu[:, 1]) * (torch.exp(mu[:, 1] * u) - 1) - g
    return (r ** 2).mean()


def train_pinn(network, steps, samples, x_boundary, mu_boundary, log=0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    network = network.to(device)
    optim = torch.optim.AdamW(network.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optim, 1000, gamma=0.75
    )
    ls = torch.linspace(x_boundary[0], x_boundary[1], samples)
    # grid on the domain
    x_inner = torch.stack(
        tuple(map(torch.flatten, torch.meshgrid(ls[1:-1], ls[1:-1])))
    ).to(device).T
    # points on the boundary
    x_bound = torch.cat(
        tuple(
            torch.stack(t)
            for t in [
                (ls, torch.zeros_like(ls)),
                (ls, torch.ones_like(ls)),
                (torch.zeros_like(ls), ls),
                (torch.zeros_like(ls), ls),
            ]
        ),
        dim=1,
    ).to(device).T
    mu_inner = torch.FloatTensor(x_inner.shape[0], 2).to(device)
    mu_bound = torch.FloatTensor(x_bound.shape[0], 2).to(device)
    for s in range(steps):
        mu_inner.uniform_(mu_boundary[0], mu_boundary[1])
        mu_bound.uniform_(mu_boundary[0], mu_boundary[1])
        optim.zero_grad()
        loss_r = residual_loss(network, x_inner, mu_inner)
        loss_b = boundary_loss(network, x_bound, mu_bound)
        loss = (loss_b + loss_r) / 2
        loss.backward()
        optim.step()
        lr_scheduler.step()
        if log and (s % log == 0):
            print(
                f"""Step {s}, lr {optim.param_groups[0]['lr']} loss {(loss_b.item() + loss_r.item()) / 2:.2}, loss_b {
                    loss_b.item():.2}, loss_r {loss_r.item():.2}"""
            )
    return network
