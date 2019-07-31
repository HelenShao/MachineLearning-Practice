import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn

import matplotlib.pyplot as plt
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import LinearAxis, Range1d

import numpy as np
import os,sys


# Define Hyperparameters
epochs = 6
batch_size = 100
num_classes = 10
learning_rate = 0.01


# Use DataLoader to pre-process image data for training and testing

# transforms to apply to the data
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# MNIST dataset
train_dataset = torchvision.datasets.MNIST('mnist_data', train=True, transform=trans, download=True)
test_dataset = torchvision.datasets.MNIST('mnist_data', train=False, transform=trans)

# Define our test and train datasets using DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

print(train_dataset)  #60,000 datapoints


# Example batch
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

example_data.shape 
"""The tensor shape is 1000 x 1 x 28 x 28 ---> 
1000 examples of 28 x 28 grayscale pixels (channel is one because no RBG)"""

# Plot some images
fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])
fig


# Computer output channgel size
def output_size(in_size, kernel_size, stride, padding):
    output = int((in_size - kernel_size + 2 * padding) / (stride)) + 1
    return(output)

def padding(in_size, out_size, kernel_size, stride):
    padding = (((out_size - 1) * stride ) + kernel_size - in_size) / 2
    return(padding)


print(64 * 7 * 7)
t = torch.rand(49,64)
t = t.reshape(t.size(0), -1)
t = t.view(1,-1)
t.size()
print(t)

# Define Model Class

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.Layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size= 5, stride = 1, padding = 2),   #Why kernel_size 5?
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.Layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = 5, stride = 1, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)     # 10 output size because 10 labels 
        
# Define how the data flows through the model: FORWARD PASS

def forward(self, x):
    x = self.Layer1(x)
    x = self.Layer2(x)
    x = x.view(-1, 1)   #Flatten out tensor to (3136 x 1) for Dropout
    x = self.drop_out(x)
    x = self.fc1(x)
    x = self.fc2(x)
    return(x)
    
# Call the model for training

model = Model()

# Create loss and optimizer function for training

criterion = nn.CrossEntropyLoss()  
"""We do not have to include Sotfmax Classifier layer in model because
CrossEntropyLoss function combines both a SoftMax activation and 
a cross entropy loss function in the same function! """

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)  #Updates Parameters

# Training the Model

#Create some printable paramters to use later on for graphing curves
total_step = len(train_loader)
total_loss = []
total_accuracy = []

Loss = np.zeros(3600)
Accuracy = np.zeros(3600)

# Training Loop

for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Run the FORWARD PASS
        prediction = model(images)
        loss = criterion(prediction, labels)   # Calculate CrossEntropyLoss between the predicted and true labels
        total_loss.append(loss.item())  #Record Loss to list
        
        #Perform Backward Propogation and Adam optimization 
        optimizer.zero_grad()  #Set initial gradient to zero
        loss.backward()        #Calculate gradient on loss
        optimizer.step()       #Update weights 
        
        #Track Accuracy
        total = labels.size(0)
        _, predicted = torch.max(prediction.data, 1)
        correct = (predicted == labels).sum().item()
        total_accuracy.append(correct / total)
        
        if (i+1) % 100  == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))
print(test_dataset)

# Testing the Model

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        test_predictions = model(images)
        _, test_predicted = torch.max(test_predictions.data, 1)
        total += labels.size(0)
        correct += (test_predicted == labels).sum().item()
        
        print('Test Accuracy of the Model on the 10000 test images: {} %'.format((correct / total) * 100))
        
# Save the model and plot
torch.save(model.state_dict(), 'conv_net_model.pt')




