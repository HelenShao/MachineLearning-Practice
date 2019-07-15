import torch
import torch.nn as nn
import numpy as np

x_train = torch.linspace(0,1,steps=500)
y_train = 3.0*x_train + 2.0

params_train = torch.tensor([3.0, 2.0])

### Linear Regression Model ###

class Model(torch.nn.Module):    # Use the Linear Torch Module
    def __init__(self):
        super(Model,self).__init__()
        self.linear = torch.nn.Linear(500,2)    # Take in 500 points, return 2 parameters
    def forward(self,x):
        params_pred = self.linear(x)
        return params_pred            # Predict the parameters

model = Model()
criterion = nn.MSELoss() #Loss Function
optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001)   #Learning rate is extremely small in this case

## Training ##

for epoch in range(3000):
    params_pred = model(y_train)                 # Input is y_train (500 points), using linear model
    loss = criterion(params_pred, params_train)  # Calculates Loss based on params_pred and params_train
    loss_tensor = torch.tensor(loss)
    print(epoch, loss_tensor.item())
    optimizer.zero_grad()
    loss.backward()         #Gradient
    optimizer.step()        #Update parameters
    print(params_pred)
