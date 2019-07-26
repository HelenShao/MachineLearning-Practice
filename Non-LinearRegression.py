import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

import matplotlib.pyplot as plt

import numpy as np


## Train Set ##
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


### Test Set ####

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


### Create Model Container ###
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
optimizer1 = torch.optim.SGD(Model.parameters(), lr = 0.008) 


### Training ###
loss_total = np.zeros(5000) # Loss for training set
error_valid_total = np.zeros(5000) # Loss for validation set

Model.train()

for epoch in range(5000):
    y_pred = Model(x_train.view(-1,1))
    loss = criterion(y_pred, y_train.view(-1,1))     #Loss function for train set
    loss_total[epoch] = loss.detach().numpy()
    optimizer1.zero_grad()
    loss.backward()
    optimizer1.step()
    
    y_validation = Model(x_testn.view(-1,1))
    error_valid = criterion(y_validation, y_testn.view(-1,1))   #Loss function for validation set
    error_valid_total[epoch] = error_valid.detach().numpy()
    
    #Print Loss for both
    loss_tensor = torch.tensor(loss)
    error_valid_tensor = torch.tensor(error_valid)
    print('epoch: ', epoch, 'loss: ', loss_tensor.item(), '  error:', error_valid_tensor.item())
    
    
# Plot Training Loss and Validation Error

plt.ylabel('error')
plt.xlabel('epochs')
plt.yscale('log')
plt.xscale('log')
plt.plot(epochs, error_valid_total, label = 'Error for Validation')
plt.plot(epochs, loss_total, label = 'Training Loss')
plt.legend(loc = 'center right')
plt.show()

# Undo Normalization for y_prediction
y_pred = y_pred * (y_max - y_min) + y_min

#Plot prediction and training set
plt.plot(torch.Tensor.numpy(x), torch.Tensor.numpy(y), label = 'Training Set')
plt.plot(torch.Tensor.numpy(x), y_pred.detach().numpy(), label = 'Prediction')
plt.legend(loc='upper left')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


#Undo Normalization for y_validation
y_validation = y_validation * (y_test_max - y_test_min) + y_test_min


#Plot prediction and validation set
plt.plot(torch.Tensor.numpy(x_test), torch.Tensor.numpy(y_test), label = 'Validation Set')
plt.plot(torch.Tensor.numpy(x_test), y_validation.detach().numpy(), label = 'Prediction')
plt.legend(loc='upper left')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
