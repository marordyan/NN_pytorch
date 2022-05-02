import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams


rcParams.update({'font.size': 18})
plt.rcParams['figure.figsize'] = [12, 12]

## Symulate the lorenz system

dt = 0.01
T = 8
t = np.arange(0,T+dt,dt)
beta = 8/3
sigma = 10
rho = 28

def lorenz_deriv(x_y_z, t0, sigma=sigma, beta=beta, rho=rho):
    x, y, z = x_y_z
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


np.random.seed(123)
x0 = -15 + 30 * np.random.random((100, 3))

x_t = np.asarray([integrate.odeint(lorenz_deriv, x0_j, t) for x0_j in x0])

## create input and output vectors to be populated from Lorenz
nn_input = np.zeros((100*(len(t)-1),3))
nn_output = np.zeros_like(nn_input)

## assign input and output to normalized Lorenz
for j in range(100):
    x, y, z = x_t[j,:,:].T
    x = (x-np.mean(x))/np.std(x)
    y = (y-np.mean(y))/np.std(y)
    z = (z - np.mean(z))/np.std(z)
    nn_input[j*(len(t)-1):(j+1)*(len(t)-1),:] = (np.array([x[:-1],y[:-1],z[:-1]])).T
    nn_output[j*(len(t)-1):(j+1)*(len(t)-1),:] = (np.array([x[1:],y[1:],z[1:]])).T

### Create the ffn

import torch
from torch import nn 
from torch import optim

model = nn.Sequential(nn.Linear(3,10), nn.ReLU(), nn.Linear(10, 10), nn.ReLU(), nn.Linear(10,10), nn.ReLU(), nn.Linear(10,3))

model.to(device = torch.device('cuda:1'))

criterion = nn.MSELoss().to(device=torch.device('cuda:1'))
optimizer = optim.SGD(model.parameters(), lr = 0.0001)

## turn intup and output into tensors

nn_input = torch.from_numpy(nn_input.astype(np.float64)).to(device=torch.device('cuda:1'))
nn_output = torch.from_numpy(nn_output.astype(np.float64)).to(device=torch.device('cuda:1'))


## train for 5000 epochs

epochs = 5000
training_loss = np.zeros(epochs)
for e in range(epochs):
    running_loss = 0
    for i in range(nn_input.shape[0]):
        optimizer.zero_grad()
        output = model(nn_input[i].float())
        loss = criterion(output,nn_output[i].float())
        loss.backward()
        optimizer.step()

        running_loss += loss.cpu().item()

    training_loss[e] = running_loss/(nn_input.shape[0])
    if not(e%100):
        print(f"Training loss: {running_loss/nn_input.shape[0]}")    
