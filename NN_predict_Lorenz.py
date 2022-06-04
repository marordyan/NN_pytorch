import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams



rcParams.update({'font.size': 18})
plt.rcParams['figure.figsize'] = [12, 12]

## Symulate the lorenz system

dt = 0.01
T = 40#80
t = np.arange(0,T+dt,dt)
beta = 8/3
sigma = 10
rho = 28
n_samples = 500

def lorenz_deriv(x_y_z, t0, sigma=sigma, beta=beta, rho=rho):
    x, y, z = x_y_z
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


np.random.seed(123)
x0 = -15 + 30 * np.random.random((n_samples, 3))

x_t = np.asarray([integrate.odeint(lorenz_deriv, x0_j, t) for x0_j in x0]) # the shape of x_t is (n_samples, time, 3)  

print(f'shape of x_t is: {x_t.shape}')

## create input and output vectors to be populated from Lorenz
nn_input = np.zeros((n_samples*(len(t)-1),3))
nn_output = np.zeros_like(nn_input)

print(f'shape of nn_input is: {nn_input.shape}')

## assign input and output to normalized Lorenz
# for j in range(n_samples):
#     x, y, z = x_t[j,:,:].T
#     x = (x-np.mean(x))/np.std(x)
#     y = (y-np.mean(y))/np.std(y)
#     z = (z - np.mean(z))/np.std(z)
#     nn_input[j*(len(t)-1):(j+1)*(len(t)-1),:] = (np.array([x[:-1],y[:-1],z[:-1]])).T
#     nn_output[j*(len(t)-1):(j+1)*(len(t)-1),:] = (np.array([x[1:],y[1:],z[1:]])).T


# do the assignment with numpy, and the normalization with the whole dataset

#x_flat = np.transpose(x_t, axes=[2,1,0]).reshape(3,-1)

#nn_input = np.transpose(x_t[:,0:-1,:], axes = [2,1,0]).reshape(-1,3)
#nn_output = np.transpose(x_t[:,1:,:], axes = [2,1,0]).reshape(-1,3)

nn_input = (x_t[:,0:-1,:]).reshape(-1,3)
nn_output = (x_t[:,1:,:]).reshape(-1,3)


nn_input = (nn_input - nn_input.mean(0))/np.std(nn_input, axis = 0)
nn_output = (nn_output - nn_output.mean(0))/np.std(nn_output,axis = 0)


plt.plot(nn_input[:len(t),0], nn_input[:len(t),1])
plt.plot(nn_input[:len(t),0], nn_input[:len(t),2])
plt.plot(nn_input[:len(t),1], nn_input[:len(t),2])

plt.show()
### Create the ffn

import torch
from torch import nn 
from torch import optim
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    my_device = 'cuda:0'
else:
    my_device = 'cpu'

model = nn.Sequential(nn.Linear(3,10), nn.ReLU(), nn.Linear(10, 10), nn.ReLU(), nn.Linear(10,10), nn.ReLU(), nn.Linear(10,3))

model.to(device = torch.device(my_device))

criterion = nn.MSELoss().to(device=torch.device(my_device))
optimizer = optim.Adam(model.parameters(), lr = 0.001)

## turn intup and output into tensors

nn_input = torch.from_numpy(nn_input.astype(np.float64)).to(device=torch.device(my_device))
nn_output = torch.from_numpy(nn_output.astype(np.float64)).to(device=torch.device(my_device))


## train for 5000 epochs

# epochs = 5000
# training_loss = np.zeros(epochs)
# batch_size = 50#*int(T/dt)#800000
# for e in range(epochs):
#     running_loss = 0
#     loader = iter(DataLoader((nn_input, nn_output), batch_size=batch_size, shuffle=False))
#     # optimizer.zero_grad()
#     # output = model(nn_input.float())
#     # loss = criterion(output, nn_output.float())
#     # loss.backward()
#     # optimizer.step()
#     idx = np.random.permutation(n_samples)
#     for i in idx:
#         optimizer.zerograd()
#         output = model(nn_input[i*int(T/dt):(i+1)*int(T/dt),:])
#     for x,y in loader:
#         # x = torch.from_numpy(x.astype(np.float64)).to(device=torch.device(my_device))
#         # y = torch.from_numpy(y.astype(np.float64)).to(device=torch.device(my_device))
#         optimizer.zero_grad()
#         #output = model(nn_input.float())
#         output = model(x.float())
#         #loss = criterion(output,nn_output.float())
#         loss = criterion(output,y.float())
#         loss.backward()
#         optimizer.step()
#         running_loss += loss 
#     training_loss[e] = running_loss/len(loader)#/(nn_input.shape[0])
#     running_loss += loss.cpu().item()
#     training_loss[e] = running_loss



#### train in random batches of 50 curves

epochs = 5000
training_loss = np.zeros(epochs)
batch_size = 50
x_norm = (x_t - x_t.mean((0,1)))/x_t.std((0,1))
for e in range(epochs):
    running_loss = 0
    for i in range(int(n_samples/batch_size)):
        idx = np.random.choice(n_samples, batch_size, replace = False)
        curr_input = torch.from_numpy((x_norm[idx,:-1,:]).reshape(-1,3)).to(device = my_device)
        curr_output = torch.from_numpy((x_norm[idx,1:,:]).reshape(-1,3)).to(device = my_device)
        optimizer.zero_grad()
        output = model(curr_input.float())
        loss = criterion(output, curr_output.float())
        loss.backward()
        optimizer.step()
        running_loss += loss
    training_loss[e] = running_loss/int(n_samples/batch_size)
    if not(e%100):
        print(f"Training loss: {training_loss[e]}")    

torch.save(model,'./lorenz_model.pth')

## test on training data after training

model.to(device = torch.device('cpu'))

i = np.random.randint(n_samples, size=1)
p = np.zeros((int(T/dt),3))
x_norm = ((x_t[i,:,:]-x_t.mean((0,1)))/x_t.std((0,1))).reshape(x_t.shape[1],x_t.shape[2])
print(f'shape of x_norm is: {x_norm.shape}')
p[0,:] = x_norm[0,:]
print(f'shape of p[0,:] is: {p[0,:].shape}')
loss_tt = np.zeros(int(T/dt)-1)
for j in range(int(T/dt)-1):
    if j < 200:
        xx = torch.from_numpy(x_norm[j,:])
    else:
        xx = torch.from_numpy(p[j,:])
    p[j+1,:] = model(xx.float()).detach().numpy()
    loss_tt[j] = np.sum((x_norm[j+1,:] - p[j+1,:])**2)/3
print(f"Total testing loss on training data is for this case is: {np.mean(loss_tt)}")


## plot predicted vs simulated data:

fig, axes = plt.subplots(1,1,subplot_kw={'projection': '3d'})
axes.plot(p[:,0],p[:,1],p[:,2], linewidth = 1)
axes.plot(x_norm[:,0],x_norm[:,1],x_norm[:,2],'k-', alpha = 0.5)
axes.view_init(18,113)
plt.savefig('./train.png')

