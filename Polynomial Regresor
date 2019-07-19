import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


### Learn Polynomial Function ###
x_train = torch.linspace(0,1,steps=500)
y_train = 3*x_train**3 + 2*x_train**2 + 7*x_train + 8


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
loss_total = np.zeros(3000)
for epoch in range(3000):
    y_pred = Model(x_train.view(-1,1))
    loss = criterion(y_pred, y_train.view(-1,1))
    loss_total[epoch] = loss.detach().numpy()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    #Print Loss
    loss_tensor = torch.tensor(loss)
    print('epoch: ', epoch, 'loss: ', loss_tensor.item())
   

# Plot loss as a function of epochs

epochs = np.arange(3000)
plt.plot(epochs, loss_total)
plt.ylabel("loss")
plt.xlabel("epochs")

### Validation ###

x_test = torch.randn(100)
y_test = 3*x_test**3 + 2*x_test**2 + 7*x_test + 8

y_prediction= (Model(x_test.view(-1,1)))
error = criterion(y_prediction, y_test.view(-1,1))
error_tensor = torch.tensor(error)
print('prediction:',y_prediction, 'error:', error_tensor)

#Plot loss as function of iterations

iterations = np.linspace(1,100,100)
plt.plot(iterations.reshape(1,-1), error.detach().numpy())
plt.show()
