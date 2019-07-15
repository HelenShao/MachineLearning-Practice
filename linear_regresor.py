import torch 
import torch.nn as nn
import numpy as np

x_train = torch.linspace(0,1,steps=500)
y_train = 3.0*x_train + 2.0

w = torch.randn(1, requires_grad=True. dtype = torch.float, device=device)
b = torch.randn(1, requires_grad=True. dtype = torch.float, device=device)

### Linear Regression Model ###

class Model(torch.nn.Module):
	def __init__(self):
        super(Model,self).__init__()
		self.linear = torch.nn.Linear(500,2)
	def forward(self,x):
		y_pred = self.linear(x)
		return y_pred
		
model = Model()

criterion = nn.MSELoss() #Loss Function
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)


### Training ##

for epoch in range(3000):
	y_pred = model(x_train.view(-1,1))  #Change the shape of the input to 1 
	loss = criterion(y_pred, y_train.view(-1,1))
   	loss_tensor = torch.tensor(loss)
   	print(epoch, loss_tensor.item())
	optimizer.zero_grad()  #Remember the gradient!
	loss.backward()        #Gradient 
	optimizer.step()       #Update parameters
   	weights = list(model.parameters())
   	print(weights)
 
### Validation ###
test_point = torch.FloatTensor([[4.0]])
print(model(test_point).data[0][0].item())
	

	
