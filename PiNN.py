import torch
import torch.nn as nn


class MLPiNN(nn.Module):
    def __init__(
        self, in_dim, out_dim, param_dim, hidden_size, hidden_layers, activation=nn.Tanh
    ):
        super(MLPiNN, self).__init__()
        assert hidden_layers > 0, "Number of hidden layers must be positive"
        assert hidden_size > 0, "Dimension of hidden layer must be non zero"
        layers = [nn.Linear(in_dim + param_dim, hidden_size), activation()]
        for layer in range(hidden_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation())
        layers.append(nn.Linear(hidden_size, out_dim))
        self.layers = nn.Sequential(*layers)
        self.loss = nn.MSELoss()

    def forward(self, x, mu):
        inp = torch.cat((x, mu), dim=1)
        return self.layers(inp)

    def boundary_loss(self, x, mu):
        return (self(x, mu) ** 2).mean()

    def residual_loss(self, x, mu):
        assert x.shape[1] == 2, "x must be 2 dimensional"
        assert mu.shape[1] == 2, "mu must be 2 dimensional"
        assert torch.all(mu[:, 1] != 0), "mu 1 must not be zero"
        u = self(x, mu).squeeze()
        div = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        dif = torch.autograd.grad(div.sum(), x, create_graph=True)[0]
        x = x.detach()
        g = 100 * torch.sin(2 * torch.pi * x[:, 0]) * torch.cos(2 * torch.pi * x[:, 1])
        r = -dif.sum(axis=1) + (mu[:, 0] / mu[:, 1]) * (torch.exp(mu[:, 1] * u) - 1) - g
        return self.loss(r, torch.zeros_like(r))


def train(network, steps, step_samples, x_boundary, mu_boundary, log=False):
    optim = torch.optim.AdamW(network.parameters(), lr=1e-2)
    for s in range(steps):
        x_inner = torch.FloatTensor(step_samples, 2).uniform_(
            x_boundary[0], x_boundary[1]
        )
        x_border = x_inner.clone().detach()
        coordinate_mask = torch.LongTensor(step_samples).random_(2)
        boundary_mask = torch.LongTensor(step_samples).random_(2)
        x_border[torch.arange(step_samples), coordinate_mask] = x_boundary[
            boundary_mask
        ]
        x_inner = torch.autograd.Variable(x_inner, requires_grad=True)
        mus = torch.FloatTensor(step_samples, 2).uniform_(
            mu_boundary[0], mu_boundary[1]
        )
        optim.zero_grad()
        loss_b = network.boundary_loss(x_border, mus)
        loss_r = network.residual_loss(x_inner, mus)
        loss = 0.5 * loss_b + 0.5 * loss_r
        if log:
            print(
                f"""Step {s}, loss {loss.item():.2}, loss_b {
                    loss_b.item():.2}, loss_r {loss_r.item():.2}"""
            )
        loss.backward()
        optim.step()



# neural network initialization
x_train = torch.tensor(np.float32(training_set))
net = MLPiNN(1, 1, 2, 10, 4)

# training the network
train(net, 100, 100, [0, 1], [0, 1], log=True)
