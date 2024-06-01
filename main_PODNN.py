import GeDiM4Py as gedim
import numpy as np
import time
from weakforms import forcing_term, ones
from FOM import newton_solver
from PODNN import *
import torch
import torch.nn as nn
from POD import pod_base

# setup lib
lib = gedim.ImportLibrary("/content/CppToPython/release/GeDiM4Py.so")
gedim.Initialize({"GeometricTolerance": 1.0e-8}, lib)

# setup problem
mesh_size = 1e-3
order = 1
domain = {
    "SquareEdge": 1.0,
    "VerticesBoundaryCondition": [1, 1, 1, 1],
    "EdgesBoundaryCondition": [1, 1, 1, 1],
    "DiscretizationType": 1,
    "MeshCellsMaximumArea": mesh_size,
}
_, mesh = gedim.CreateDomainSquare(domain, lib)

discreteSpace = {"Order": order, "Type": 1, "BoundaryConditionsType": [1, 2]}
problem_data, dofs, strongs = gedim.Discretize(discreteSpace, lib)
n_dofs = dofs.shape[1]

# snapshots
n_train = 10
train_set = np.random.uniform(0.1, 1, size=(n_train, 2))

# extract basis
print(f"Computing pod basis...")
basis_time = time.process_time()
basis, inner_product, snapshots = pod_base(lib, problem_data, train_set)
basis_time = time.process_time() - basis_time
print(f"Computed basis in {basis_time:.2}s")

# snapshotProjector to get the training set in the reduced space
y_train = snapshotProjector(snapshots, basis, inner_product)
x_train = torch.tensor(np.float32(train_set))

# Neural network 

out_dim = y_train.shape[1] #da fare check sulle dimensioni
in_dim = x_train.shape[1]
torch.set_default_dtype(torch.float32)
net = podNN(out_dim, in_dim, hidden_size = 100 , hidden_layers = 5, activation=nn.Tanh)
lossFunction = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
epoch_max = 500000
epoch = 0
tol = 1e-5
loss = 1.

while loss >= tol and epoch < epoch_max:  #training
  epoch = epoch + 1
  optimizer.zero_grad()
          
  # compute output and loss function
  output = net(x_train) #inserire numero with e depth del nn
  loss = lossFunction(output, y_train)
  
  if epoch >= 20000:
    optimizer.param_groups[0]['lr'] = 0.0001  
  
  loss.backward() #compute the gradients
 
  optimizer.step()  # optimizer update the weights
  
  if epoch % 200 == 199:
    print("epoch", epoch, 'loss', loss.item(), 'lr', optimizer.param_groups[0]['lr'] )


  