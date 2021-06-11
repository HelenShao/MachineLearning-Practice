############################################ Custom Modules in Pytorch ########################################
# References:
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-custom-nn-modules
# https://auro-227.medium.com/writing-a-custom-layer-in-pytorch-14ab6ac94b77

# Introduction:
# Defining custom Modules (layers) by subclassing nn.Module and defining a forward function which receives 
# *input* Tensors and produces output Tensors using other modules or other autograd operations on Tensors.
# Use nn.Parameter() to create weights and biases that can be updated during training

# Example 1:
import torch
import math

class Polynomial3(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate (initialize) four parameters and assign them as members (instance vars).
        super().__init__() allows for the inheritance of nn.Module class
        """
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))
        
        # You can define other layers or sequences of layers here.
        # Define sequential containers of layers using nn.Sequential() - see next example

    def forward(self, x):
        """
        In the forward function we accept a Tensor of *input data* and we must return
        a Tensor of output data. 
        You can use Modules (layers) defined in the constructor as
        well as arbitrary operators on Tensors.
        You can also use other Torch functions from torch.nn.functional (F)
        """
        return self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3

    def string(self):
        """
        Just like any class in Python, you can also define custom method on PyTorch modules
        This method returns the string representation of the model, including all of its weights
        and parameters.
        (tensor.item() returns a Python number from a tensor)
        """
        return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3'

# Create Tensors to hold input and outputs.
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# Construct our model by creating an instance of the class defined above
model = Polynomial3()

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters (defined 
# with torch.nn.Parameter) which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
for t in range(2000):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'Result: {model.string()}')

# Example 2: Custom Layer used in Neural Net
# Inherit properties from nn.Module class
# Use nn.Parameter to get trainable variables
# Initialize weights and biases in the constructors as nn.Parameter Tensors - torch.Tensor simply defines the variable
# as type tensor. It is an empty tensor with defined shape
# reset_parameters = initializing the values of the weights using kaiming uniform
# Forward function multiplies the input of the model with the weights in a 1-1 fashion
# Define string representation method. You can also use list(model.children())[0].weight to get the values of the 
# weights of this first layer

class FirstLayer(nn.Module):
    def __init__(self, n_features, bias = True): 
        super(FirstLayer, self).__init__()
        self.n_features = n_features
        self.weight = nn.Parameter(torch.Tensor(1, n_features))  # Define weight as a parameter of the layer to be trained
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) #?
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            
    def forward(self, input):
        output = torch.mul(input, self.weight) # Multiply the input with the weight to get output
        if self.bias is not None:
            output += self.bias
        return output
    
    def toString(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.n_features, self.n_features, self.bias is not None
        )
      
# Example 3: UNET
# First, define the sequence of layers to be used later to build the UNET - the double convolutional + double activation func
# In the UNet class, inherit nn.Module and initialize the layers in the down and up sections of the UNET
# In the forward function, pass the input through the layers defined in the constructor.

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels,  out_channels, 3, padding=1),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.LeakyReLU(inplace=True)
    )   

class UNet(nn.Module):

    def __init__(self):
        super(UNet,self).__init__()
                
        self.dconv_down1 = double_conv(2, 1)
        self.dconv_down2 = double_conv(1, 2)
        self.dconv_down3 = double_conv(2, 4)
        self.dconv_down4 = double_conv(4, 8)        

        self.maxpool = nn.MaxPool2d(2)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', 
                                    align_corners=True)        
        
        self.dconv_up3 = double_conv(8 + 4, 8)     #concatenated with conv_down3 
        self.dconv_up2 = double_conv(8 + 2, 4)      #concatenated with conv_down2
        self.dconv_up1 = double_conv(4 + 1 + 2, 2)   #concatenated with conv_down1 and 2 inputs
        self.conv_last = nn.Conv2d(2, 1, 1) 
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x1    = self.maxpool(conv1)
        
        conv2 = self.dconv_down2(x1)
        x2    = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x2)
        x3    = self.maxpool(conv3)   
        
        x4 = self.dconv_down4(x3)
        
        x5 = self.upsample(x4)  
        x6 = torch.cat([x5, conv3], dim=1)
        x7 = self.dconv_up3(x6)

        x8 = self.upsample(x7)        
        x9 = torch.cat([x8, conv2], dim=1)       
        x10 = self.dconv_up2(x9)

        x11 = self.upsample(x10)        
        x12 = torch.cat([x11, conv1, x], dim=1)           
        x13 = self.dconv_up1(x12)
        out = self.conv_last(x13)
        
        return out
