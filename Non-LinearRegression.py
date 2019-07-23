import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

import matplotlib.pyplot as plt

import numpy as np


## Function ##
x = torch.linspace(0,20,5000)
y = 1 + 2*x + 3/(1+x) + torch.sin(x)

plt.plot(torch.Tensor.numpy(x) , torch.Tensor.numpy(y))
plt.ylabel('Y')
plt.xlabel('X')
plt.show()

#Normalize Data
x_min = torch.min(x)
x_max = torch.max(x)

y_min = torch.min(y)
y_max = torch.max(y)

x_train= (x - x_min)/(x_max - x_min)
y_train= (y - y_min)/(y_max - y_min)

plt.plot(torch.Tensor.numpy(x_train), torch.Tensor.numpy(y_train))
plt.xlabel('X_Train')
plt.ylabel('Y_Train')
plt.show()


# Create Model Container
Model = nn.Sequential()

# Define Layers
Linear_Layer = nn.Linear(1,100)       # Add many neuronsin hidden layers
Relu1 = nn.ReLU(inplace=False)
Linear_Layer2 = nn.Linear(100,200)
Relu2 = nn.ReLU(inplace=False)
Linear_Layer3 = nn.Linear(200,1)


# Add layers to model

Model.add_module("Lin1", Linear_Layer)
Model.add_module("Relu1", Relu1)              #Activation for first linear
Model.add_module("Lin2", Linear_Layer2)
Model.add_module("Relu2", Relu2)              #Activation for second linear
Model.add_module("Lin3", Linear_Layer3)

criterion = nn.MSELoss()       #Loss Function      
optimizer = torch.optim.SGD(Model.parameters(), lr = 0.0001) 

### Training ###

loss_total = np.zeros(5000)
Model.train()

for epoch in range(5000):
    y_pred = Model(x_train.view(-1,1))
    loss = criterion(y_pred, y_train.view(-1,1))
    loss_total[epoch] = loss.detach().numpy()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    #Print Loss
    loss_tensor = torch.tensor(loss)
    print('epoch: ', epoch, 'loss: ', loss_tensor.item())


### Plot loss as a function of epochs ###

epochs = np.arange(5000)
plt.plot(epochs, loss_total)
plt.ylabel("loss")
plt.xlabel("epochs")
plt.yscale('log')
plt.xscale('log')

# Test Set

x_test = torch.randn(5000)
y_test = 1 + 2*x_test.view(-1,1) + 3/(1+x_test.view(-1,1)) + torch.sin(x_test.view(-1,1))

#Normalize Data
x_test_min = torch.min(x_test)
x_test_max = torch.max(x_test)

y_test_min = torch.min(y_test)
y_test_max = torch.max(y_test)


x_testn= (x_test - x_test_min)/(x_test_max - x_test_min)
y_testn= (y_test - y_test_min)/(y_test_max - y_test_min)

print(y_testn)
plt.scatter(torch.Tensor.numpy(x_testn), torch.Tensor.numpy(y_testn))
plt.show()

### Validation ###

Model.eval() #Set model into evaluation mode

error_total = np.zeros(5000)

for epoch in range(5000):
    y_prediction = Model(x_testn.view(-1,1))
    error = criterion(y_prediction, y_testn.view(-1,1))
    error_total[epoch] = error.detach().numpy()
    optimizer.zero_grad()
    error.backward()
    optimizer.step()

print(y_prediction, error_total)
    
epochs = np.arange(5000)
plt.plot(epochs, error_total)
plt.show()

# Plot Training Loss and Validation Error

plt.ylabel('error')
plt.xlabel('epochs')
plt.yscale('log')
plt.xscale('log')
plt.plot(epochs, error_total)
plt.plot(epochs, loss_total)
plt.show()
