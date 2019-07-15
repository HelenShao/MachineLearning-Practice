
import torch 
import torch.nn as nn
import numpy as np


x_train = torch.linspace(0,1,steps=500)
y_train + torch.linspace(0,1,steps=500)

print(y_train.shape)

### Linear Regression Model ###

class Model(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.linear = torch.nn.Linear(1,1)
	def forward(self,x):
		y_pred = self.linear(x)
		return y_pred
		
model = Model()

criterion = nn.MSELoss() #Loss Function
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

### Training ##

for epoch in range(100):
	y_pred = model(x_train)
	print(y_pred.shape)
	print(y_train.shape)
	loss = criterion(y_pred, y_train)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	

	