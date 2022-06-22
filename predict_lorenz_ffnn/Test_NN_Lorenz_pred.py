import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from scipy import integrate
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams

# model = nn.Sequential(nn.Linear(3,10), nn.ReLU(), nn.Linear(10,10), nn.ReLU(), nn.Linear(10,10), nn.ReLU(), nn.Linear(10,3))
# model.load_state_dict(torch.load('./lorenz_model.pth'))
model = torch.load('./lorenz_model.pth')
print(model.eval())
model.to(device = torch.device('cpu'))

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


np.random.seed(126)
x0 = -15 + 30 * np.random.random((n_samples, 3))

x_t = np.asarray([integrate.odeint(lorenz_deriv, x0_j, t) for x0_j in x0]) # the shape of x_t is (n_samples, time, 3)  

print(f'shape of x_t is: {x_t.shape}')
## test on training data after training


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

#plt.plot(loss_tt)

## plot predicted vs simulated data:

# fig, axes = plt.subplots(1,1,subplot_kw={'projection': '3d'})
# axes.plot(p[:,0],p[:,1],p[:,2], linewidth = 1)
# axes.plot(x_norm[:,0],x_norm[:,1],x_norm[:,2], 'k-', alpha = 0.5)
# axes.view_init(18,113)
# plt.show()


x_norm = ((x_t[:,:,:]-x_t.mean((0,1)))/x_t.std((0,1)))
lt = np.zeros(n_samples)
for i in range(n_samples):
    running_loss = 0
    p = np.zeros((int(T/dt),3))
    p[0,:] = x_norm[i,0,:].reshape(-1)
    
    for j in range(int(T/dt)-1):
        if j < 200:
            xx = torch.from_numpy(x_norm[i,j,:])
        else:
            xx = torch.from_numpy(p[j,:])

        p[j+1,:] = model(xx.float()).detach().numpy()
        if np.sum((x_norm[i,j+1,:] - p[j+1,:])**2)/3 > 0.1:
            lt[i] = (j-200)*dt
            break

plt.plot(lt)
plt.show()