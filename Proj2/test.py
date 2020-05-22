#!/usr/bin/env python
# coding: utf-8

# In[1]:


print('###################################################################')
print('')
print("Project 2")
print("Alexander Rusnak - SCIPER #: 309939 - May 2020")
print("EE 559 - Deep Learning, Professor Francois Fleuret")
print("")
print("")
print("Below I will demonstrate modules, each contextualized in the scope of the project and described as they come up. I will then train and evaluate the multiple combinations of those modules applied to the same basic architecture prescribed by the project assignment.")
print("")
print("")
print('###################################################################')
print('')


# In[2]:


# Importing torch and math
print('Turning off autograd after importing only torch and math...')
print('')
import torch
import math
from math import sqrt
# Make sure autograd is off
print(torch.set_grad_enabled(False))
print('')
print('###################################################################')
print('')


# In[3]:


# Function for generating datasets
''' Generates  a  training  and  a  test  set  of  1,000  points  sampled  uniformly 
in  [0,1]**2,  each  with  a label 0 if outside the disk centered at (0.5,0.5) of radius 
1/√2π, and 1 inside'''

def generate_data(N):
        rand_tensor = torch.rand((N, 2))
        labels = torch.empty(1000, 1)
        x = abs(0.5-rand_tensor[:,0])
        y = abs(0.5-rand_tensor[:,1])
        for i in range(N):
            if(sqrt(y[i]**2 + x[i]**2)>(1/sqrt(2*math.pi))):
                labels[i] = 0
            else: 
                labels[i] = 1
        return rand_tensor, labels
    


# In[4]:


# Generate training dataset
train_data, train_target  = generate_data(1000)
# Generate test dataset
test_data, test_target  = generate_data(1000)


# In[5]:


# ------------------ Start of the Framework ------------------
print('------------------ Start of the Framework ------------------')


# In[6]:


# Defining various activation functions with forward and backward passes

# Class for reLu activation
class relu(object):
    def  forward(self , X):
        for i in range(len(X)):
            if (X[i]<0.0):
                X[i] = 0.0
        return X
    def backward(self , X):
        for j in range(len(X)):
            if (X[j]<0.0):
                X[j]=0.0
            else: 
                X[j]= 1.0
        return X
    
# Class for tanh activation
class tanh(object):
    def  forward(self , X):
        return X.tanh()
    def backward(self , x):
        return 4 * (x.exp() + x.mul(-1).exp()).pow(-2)

# Class for sigmoid activation
class sigmoid(object):
    def forward(self , X):
        S = 1 / (1 + (-X).exp())
        return S
    
    def backward(self , X):
        S = self.forward(X)
        dS = S * (1 - S)
        return dS


# In[7]:


# Defining a class for dropout of nodes, relies on a randomly generated
# number to determine if a node should be dropped for that iteration

class dropout(object):
    def __init__(self, rate):
        invRate = rate * 2.5
        self.rate = 2.5 - invRate
    def forward(self , a, shape):
        for i in range(shape):
                c = torch.randn(1)
                if(c>self.rate or c<-self.rate):
                    a[i] == 0.0
        return a
    


# In[8]:


# Defining loss functions of mse, dloss, and loss

def dloss(v, t):
    return 2 * (v - t)

def loss(v, t):
    return (v - t).pow(2).sum()

def mse(y_pred, y):
    return ((y - y_pred)**2).mean()


# In[9]:


# Defining functions of xavier initialization of the weights.
# Normal initialization was running into vanishing/exploding gradients problem, 
# so I implemented this to help ameliorate the problem

def xavier(m,h):
    return torch.empty(m,h).uniform_(-1,1)*math.sqrt(6./(m+h))
    
def xavierInit(shapeX, shapeY, bias):
    x = torch.randn(shapeX, shapeY)
    a = xavier(shapeX, shapeY)
    if (shapeX == shapeY):
        x = (a.mm(x)).tanh()
    else: 
        x = (a * x).tanh()
    # Collapse values to 1 dimensional array for bias 
    if (bias):
        x = x[0,:]
    return x


# In[10]:


# Implementing the linear module, aka a fully connected layer.
# This module has the functions: init, forward (for forward pass), backward (for backward pass)
# zero grads (to reset the gradients), gradStep (to apply the gradients to the weights),
# and param (returns the parameters of the layer)

class FCLayer(object):
    
    # On initialization, it takes the input size of the layer, the number of units, the activation function,
    # the type of weight initialization, and the epsilon (to be used if the weight initialization)
    # is normal)
    
    def __init__(self, inputSize, units, acti, wInit, epsilon=1e-6):
        
        # Initializes weight by Xavier
        if(wInit=='xavier'):
            self.weight = xavierInit(units, inputSize, False)
            self.bias = xavierInit(1, units, True)
            
        # Initializes weight by Normal
        elif (wInit=='normal'):
            self.weight = torch.empty(units, inputSize).normal_(0, epsilon)
            self.bias = torch.empty(units).normal_(0, epsilon)
        
        # Initializes the grads as empty for backprop
        self.dW = torch.empty(self.weight.size())
        self.dB = torch.empty(self.bias.size())
        
        # Sets the activation function to be used for this layer
        if (acti=='relu'):
            self.activation = relu()
        elif (acti=='tanh'):
            self.activation = tanh()
        elif (acti=='sigmoid'):
            self.activation = sigmoid()
            
    # Forward pass of the layer, accepts the output of previous layer (or the input data)
    def  forward(self , A0):
        # Set the param for input as A0
        self.A0 = A0
        
        # Apply weights and bias to the input to get Z
        self.Z = self.weight.mv(self.A0) + self.bias
        
        #Apply the activation function to get the output
        self.A = self.activation.forward(self.Z)
        
        #return the output
        return self.A
    
    # Backward pass for the layer, accepts the loss data or the output of the backward pass
    # from the layer upstream from it (i.e. if this layer is 'l', accepts from layer l+1)
    def backward(self, dA_prev):
        
        # Set parameter for input 
        self.dA_curr = dA_prev
        
        # Calculate the derivative of Z by passing the bias backwards through the 
        # activation function and multiplying it by the input of the backward pass
        self.dZ = self.activation.backward(self.bias) * self.dA_curr
        
        # Calculate the gradients with respect to the weights
        self.dA = self.weight.t().mv(self.dZ)
        
        # Add the gradient with respect to the weights of the input from the forward pass
        # to dW for the gradient update step
        self.dW.add_(self.dZ.view(-1, 1).mm(self.A0.view(1, -1)))
        
        # Add gradients to dB for use during the gradient update step
        self.dB.add_(self.dZ)
         
        # Return dA to be passed to the downstream layer
        return self.dA
    
    # Function to zero the gradients 
    def zeroGrads(self):
        
        self.dW.zero_()
        self.dB.zero_()
    
    # Function to apply gradients to the weights and bias 
    def gradStep(self, eta):
    
        self.weight = self.weight - eta * self.dW
        self.bias = self.bias - eta * self.dB
    
    # Function that returns the parameters of the layer
    def param(self):
        
        # Weights and gradients for weights
        pairW = [self.weight, self.dW]
        
        # Bias and gradients for bias
        pairB = [self.bias, self.dB]
        
        # Z value and derivative of Z
        pairZ = [self.Z, self.dZ]
        
        # Input, output, and derivative wrt weights of the backpass input (date passed by loss or upstream layer)
        tripA = [self.A0, self.A, self.dA]
        
        # Return params
        return pairW, pairB, pairZ, tripA
    


# In[11]:


print("")
print('###################################################################')
print('')
print('In the source code I defined multiple modules/functions:')
print('')
print('   - Activation functions with forward and backward pass functions for relu, tanh, and sigmoid. ')
print('   - Loss functions of MSE, Dloss, and loss.')
print('   - Xavier weight initialization.')
print('   - Dropout Layer')
print('   - Fully Connected Layer (Linear)')
print('   - ModelClass (Sequential) - which can be called to combine these elements into a trainable model.')
print('   - Evaluate - which tests the models on the test data.')
print('')
print('I also generated train and test data as prescribed by the project sheet.')
print("")
print('For further details/documentation, look in the source code for my comments.')
print('')
print('###################################################################')
print('')
print('In the following subsection, I will run a mock model using just the various lower level modules (i.e. not modelClass) to demonstrate their effect. ')
print('I will define a set of four layers with the shape /2/25/25/1/, each with relu activation besides the last layer which uses sigmoid.')
print('Each layer will have its weights initialized with Xavier, a learning rate of 0.001, and no dropout. The loss is Dloss.')
print('')
print('I will run through one forward pass, one backward pass, one application of the gradients to the weights, and one last forward pass to show the effect')
print('All steps will be using a single piece of corresponding input and label data.')
print('')
print('-------------------------------------------------------------------')
print('')


# In[12]:


# These functions and this code are used only for example of the functionality of the modules
# They do not have any functional use in the framework itself, just serve as demostrations.

# This function zeros the gradients and runs a forward pass on the demo layers
def forwardStepTest(index):
    f1a.zeroGrads()
    f2a.zeroGrads()
    f3a.zeroGrads()
    f4a.zeroGrads()
    print('Training input data: ' + str(train_data[index]))
    print('')
    g = f1a.forward(train_data[index])
    g1 = f2a.forward(g)
    g2 = f3a.forward(g1)
    g3 = f4a.forward(g2)
    print('Output of 4 layers: ' + str(g3))
    print('')
    return g3

# This function runs a backward pass on the demo layers
def backStepTest(g, index):
    print('Training target: ' + str(train_target[index]))
    print('')
    cost = dloss(g, train_target[index])
    print('Cost function output: ' + str(cost))
    print('')
    print('Running backward pass...')
    print('')
    k = f4a.backward(cost)
    k = f3a.backward(k)
    k = f2a.backward(k)
    k = f1a.backward(k)

# This function applies the gradients to the weights of the demo layers
def gradStepTest(eta):
    print('Applying gradients...')
    print('')
    f1a.gradStep(eta)
    f2a.gradStep(eta)
    f3a.gradStep(eta)
    f4a.gradStep(eta)
    
# Defining the demo layers themselves
acti = 'relu'
wInit = 'xavier'
f1a = FCLayer( 2, 2, acti, wInit, 0.05)
f2a = FCLayer( 2, 25, acti, wInit, 0.05)
f3a = FCLayer( 25, 25, acti, wInit, 0.05)
f4a = FCLayer( 25, 1, 'sigmoid', wInit, 0.05)

# Index is just which piece of data from the 
index = 10
learningRate = 0.001

# Run the forward pass
demoOut1 = forwardStepTest(index)

# Run the backwards pass
backStepTest(demoOut1, index)

# Apply the gradients
gradStepTest(learningRate)
print('2nd forward pass with same input data to demonstrate change in weighting: ')
print('')
# Run the forward pass again to demonstrate the gradient
demoOut2 = forwardStepTest(index)
print('')
print('Difference between predicted values: ' + str(demoOut1 - demoOut2) + ' closer to target.')
print('')
print('###################################################################')
print('')


# In[13]:


print('''In between these steps I defined the module modelClass which represents the sequential model, (i.e the class builds a model with multiple layers). It has the functions init, forward, backward, zeroAllGrads, gradStepAll, train, and params

On initialization the model accepts inputs: inputSize, outputSize, units (applied to the central layers), acti (activation function type), wInit (weight initialization type), loss which determines the loss function to be used, epsilon (used if the initialization type is normal), and dropRate (what percentage of nodes should be dropped in dropout layers). ''')

print('')
print('For further details/documentation, look in the source code for my comments.')


# In[14]:


# This module defines the sequential model, (i.e the class builds a model with multiple 
# layers). It has the functions init, forward, backward, zeroAllGrads, gradStepAll, train, and params

class modelClass(object):
    
    # On initialization the model accepts inputs: inputSize, outputSize, units (applied to
    # the central layers), acti (activation function type), wInit (weight initialization 
    # type), loss which determines the loss function to be used, epsilon (used if the 
    # initialization type is normal), and dropRate (what percentage of nodes should be 
    # dropped in dropout layers). 
    def __init__(self, inputSize, outputSize, units, acti, wInit, loss, epsilon=1e-6, dropRate=1.7):
        # Defines five fuly connected layers with input specifications 
        self.f1 = FCLayer( inputSize, inputSize, acti, wInit, epsilon)
        self.f2 = FCLayer( inputSize, units, acti, wInit, epsilon)
        self.f3 = FCLayer( units, units, acti, wInit, epsilon)
        self.f4 = FCLayer( units, units, acti, wInit, epsilon)
        # The last layers always has sigmoid activation
        self.f5 = FCLayer( units, outputSize, 'sigmoid', wInit, epsilon)
        
        # Defines the dropout object with the specified rate
        self.dropR = dropout(dropRate)
        
        # Defines the loss function
        self.loss = loss
    
    # Runs a forward pass on the model, accepts x (input), and whether or not to use dropout
    def forwardPass(self , x, dropout):
        
        # Input layer
        self.A = self.f1.forward(x)
        
        # Layer 2
        self.A = self.f2.forward(self.A)
        
        # Dropout layer (conditional)
        if(dropout):
            self.A = self.dropR.forward(self.A, self.A.shape[0])
        
        #Layer 3
        self.A = self.f3.forward(self.A)
        
        # Dropout layer (conditional)
        if(dropout):
            self.A = self.dropR.forward(self.A, self.A.shape[0])
            
        #Layer 4
        self.A = self.f4.forward(self.A)
        
        # Dropout layer (conditional)
        if(dropout):
            self.A = self.dropR.forward(self.A, self.A.shape[0])
        
        # Output Layer
        self.A = self.f5.forward(self.A)
        
        # Return output
        return self.A
    
    # Runs a backward pass on the model 
    def backwardPass(self , dA_F):
        
        # Runs backward on each layer
        self.dA = self.f5.backward(dA_F)
        self.dA = self.f4.backward(self.dA)
        self.dA = self.f3.backward(self.dA)
        self.dA = self.f2.backward(self.dA)
        self.dA = self.f1.backward(self.dA)
        
        # Returns no output
        
    # Zeros the gradeints of each layer
    def zeroAllGrads(self):
        self.f1.zeroGrads()
        self.f2.zeroGrads()
        self.f3.zeroGrads()
        self.f4.zeroGrads()
        self.f5.zeroGrads()
        
        # No output
        
    # Applies gradients to all layers, accepts learning rate (eta)
    def gradStepAll(self, eta):
        self.f1.gradStep(eta)
        self.f2.gradStep(eta)
        self.f3.gradStep(eta)
        self.f4.gradStep(eta)
        self.f5.gradStep(eta)
        
    # No output
    
    # Runs the training operation on the given model. Accepts training data, training target, # of epochs, and learning rate (eta)
    def train(self, train_data, train_target, epochs, eta):
        
        # Runs for designated number of epochs
        for epoch in range(epochs):
            
                # Sets number of training errors and accumulated loss to 0
                nb_train_errors = 0
                acc_loss = 0
                correct = 0
                
                #Zeros gradients before training Epoch
                self.zeroAllGrads()
                
                # Runs for amount of pieces of data (each batch is only one item), 
                # i.e. 1000 iterations
                for batch in range(len(train_target)):
                    
                    # Gets input and target data at each step
                    X_batch = train_data[batch]
                    y_batch = train_target[batch]
                    
                    # Passed the data through the model with dropout 
                    outA = self.forwardPass(train_data[batch], True)
                
                    # Calculates the loss using the defined method
                    if(self.loss == 'dloss'):
                        cost = dloss(outA, train_target[batch])
                    elif(self.loss == 'mse'):
                        cost = mse(outA, train_target[batch])
                    
                    # Runs the backward pass with the cost
                    self.backwardPass(cost)
                    
                    # Count as wrong prediction by loss if the loss is greater than 
                    # 0.5 or less than -0.5
                    if (cost > 0.5 or cost < -0.5):
                        nb_train_errors = nb_train_errors + 1
                    
                    # Count as a correct prediction if the output value is above 0.5 for target 1
                    # or below 0.5 for target 0
                    if ((outA > 0.5 and test_target[batch] == 1) or (outA < 0.5 and test_target[batch] ==0)):
                        correct = correct + 1
                    
                    # Adds the loss to the accumulated loss at each batch step 
                    acc_loss = acc_loss + loss(outA, train_target[batch])
                # Applies accumulated gradients to the weights
                self.gradStepAll(eta) 
                
                #Print epoch data at first, then every 10 epochs
                if((epoch%10==0) or ((epoch%(epochs-1)==0))):
                    print('')
                    print('Epoch: {:d} Accumulated Loss: {:.02f} Loss Error Rate: {:.02f}%'.format(epoch,acc_loss, (100 * nb_train_errors) / len(train_data)) +  ' Correct Prediction Rate:' +  str(correct/10)+'%')
                    print('Batch: ' + str(batch) + '/' + str(len(train_target)) + ' Label: '+str(train_target[batch]) +' Model Output: '+ str(outA) + ' Batch Cost: ' + str(cost))
                    print('')
                    print('.........')
                    
# Param function returns the layers, the loss, the droprate, the last output, and the last loss input for backprop
def params(self):
    return self.f1, self.f2, self.f3, self.f4, self.f5, self.loss, self.dropR, self.A, self.dA_F


# In[15]:

print('')
print('###################################################################')
print('')
print('Instantiating the various models to demonstrate....')
print('')
print('###################################################################')

# Defining multiple models to compare different setups

# Input data size
inputSize = 2

# Output data size
outputSize = 1

# The number of units in the middle layers of the network (default = 25)
units = 25

# The epsilon value for models using normal 
epsilon = 0.3

# The dropout Rate for the models
dropRate = .05

# Create all the models with different specifications

# All models have the same input size, output size, epsilon, 
# (if they have normal initialization), and dropout rate. 

# Each model represents a unique combination of activation function (relu or tanh),
# weight initialization (normal or xavier), and loss function (mse or dloss)

modelTanhNormalDloss = modelClass(inputSize, outputSize, units, 'relu', 'normal', 'dloss', epsilon, dropRate)

modelTanhNormalMse = modelClass(inputSize, outputSize, units, 'relu', 'normal', 'mse', epsilon, dropRate)

modelReluNormalDloss = modelClass(inputSize, outputSize, units, 'tanh', 'normal', 'dloss', epsilon, dropRate)

modelReluNormalMse = modelClass(inputSize, outputSize, units, 'tanh', 'normal', 'mse', epsilon, dropRate)

modelTanhXavierDloss = modelClass(inputSize, outputSize, units, 'tanh', 'xavier', 'dloss', dropRate)

modelTanhXavierMse = modelClass(inputSize, outputSize, units, 'tanh', 'xavier', 'mse', dropRate)

modelReluXavierDloss = modelClass(inputSize, outputSize, units, 'relu', 'xavier', 'dloss', dropRate)

modelReluXavierMse = modelClass(inputSize, outputSize, units, 'relu', 'xavier', 'mse', dropRate)


# In[16]:


print('')
print('Training different models for comparison of modules/ effectiveness:')
print('')

print('All models have the same input size, output size, epsilon, (if they have normal initialization), and dropout rate. ')
print('All models have the correct number of units at each layer as specified in the assignment document. /2/25/25/25/1/')
print('All models receive identical train and test data.')
print('All models train for 20 epochs with a learning rate of 0.000085.')
print('All models have sigmoid activation on their last layer.')
print('')

print('Each model is different in its combination of activation function, weight initialization, and loss function.')
print('The options are, respectively, either Tanh or Relu activation, either Normal or Xavier init, and either MSE or Dloss loss function)')
print('')
print('###################################################################')
print('')
# Number of epochs
epochs = 20

# Learning Rate
learningRate = 0.000085

print('Model Tanh Normal Dloss')
print('.........')
modelTanhNormalDloss.train(train_data, train_target, epochs, learningRate)
print('-------------------------------------------------------------------')
print('')

print('Model Tanh Normal MSE')
print('.........')
modelTanhNormalMse.train(train_data, train_target, epochs, learningRate)
print('-------------------------------------------------------------------')
print('')

print('Model Relu Normal Dloss')
print('.........')
modelReluNormalDloss.train(train_data, train_target, epochs, learningRate)
print('-------------------------------------------------------------------')
print('')

print('Model Relu Normal MSE')
print('.........')
modelReluNormalMse.train(train_data, train_target, epochs, learningRate)
print('-------------------------------------------------------------------')
print('')

print('Model Tanh Xavier Dloss')
print('.........')
modelTanhXavierDloss.train(train_data, train_target, epochs, learningRate)
print('-------------------------------------------------------------------')
print('')

print('Model Tanh Xavier MSE')
print('.........')
modelTanhXavierMse.train(train_data, train_target, epochs, learningRate)
print('-------------------------------------------------------------------')
print('')

print('Model Relu Xavier Dloss')
print('.........')
modelReluXavierDloss.train(train_data, train_target, epochs, learningRate)
print('-------------------------------------------------------------------')
print('')

print('Model Relu Xavier MSE')
print('.........')
modelReluXavierMse.train(train_data, train_target, epochs, learningRate)
print('-------------------------------------------------------------------')
print('')
print('###################################################################')


# In[17]:


# This function evaluates a model on the test data
def evaluateModel(model):
    
    # Tracks number of correct predictions, number of loss errors (loss greater than 0.5 from 0),
    # and accumulated loss.
    correct = 0
    nb_train_errors = 0
    acc_loss = 0
    
    # Steps through test data
    for i in range(len(test_data)):   
        outA = model.forwardPass(test_data[i], False)
        
        # Count as wrong prediction by loss if the loss is greater than 
        # 0.5 or less than -0.5
        cost = dloss(outA, test_target[i])
        if (cost > 0.5 or cost < -0.5):
            nb_train_errors = nb_train_errors + 1
                    
        # Count as a correct prediction if the output value is above 0.5 for target 1
        # or below 0.5 for target 0
        if ((outA > 0.5 and test_target[i] == 1) or (outA < 0.5 and test_target[i] ==0)):
            correct = correct + 1
                    
        # Adds the loss to the accumulated loss 
        acc_loss = acc_loss + loss(outA, test_target[i])
        
    # Print Information
    print('----------------------------------------------')
    print('')
    print('Accumulated Loss: {:.02f} Loss Error Rate: {:.02f}%'.format(acc_loss, (100 * nb_train_errors) / len(train_data)) +  ' Correct Prediction Rate:' +  str(correct/10)+'%')
    print('Batch:' + str(i) + '/' + str(len(train_target)) + ' Label: '+str(train_target[i]) +' Model Output: '+ str(outA) + ' Batch Cost: ' + str(cost))
    print('')
    
    # Return number of correct predictions
    out = str(correct/10)
    return out
    


# In[18]:


# Evaluates all the models based on the test data
print('###################################################################')
print('')
print('Evaluate each model on the test data:')
print('')
print('-------------------------------------------------------------------')
print('')

print('Model Tanh Normal Dloss')
TND = evaluateModel(modelTanhNormalDloss)
print('')
print('-------------------------------------------------------------------')
print('')

print('Model Tanh Normal Mse')
TNM = evaluateModel(modelTanhNormalMse)
print('')
print('-------------------------------------------------------------------')
print('')

print('Model Relu Normal Dloss')
RND = evaluateModel(modelReluNormalDloss)
print('')
print('-------------------------------------------------------------------')
print('')

print('Model Relu Normal Mse')
RNM = evaluateModel(modelReluNormalMse)
print('')
print('-------------------------------------------------------------------')
print('')

print('Model Tanh Xavier Dloss')
TXD = evaluateModel(modelTanhXavierDloss)
print('')
print('-------------------------------------------------------------------')
print('')

print('Model Tanh Xavier Mse')
TXM = evaluateModel(modelTanhXavierMse)
print('')
print('-------------------------------------------------------------------')
print('')

print('Model Relu Xavier Dloss')
RXD = evaluateModel(modelReluXavierDloss)
print('')
print('-------------------------------------------------------------------')
print('')

print('Model Relu Xavier MSE')
RXM = evaluateModel(modelReluXavierMse)
print('')
print('-------------------------------------------------------------------')
print('')
print('')

print('Comparative Correct Prediction Rate:')
print('')
print('|------------------------------------------------------|')
print('| TND  | TNM  | RND  | RNM  | TXD  | TXM  | RXD  | RXM |')
print('|------------------------------------------------------|')
print('| ' + TND + ' | ' + TNM + ' | ' + RND + ' | ' + RNM + ' | ' + TXD + ' | ' + TXM + ' | ' + RXD + ' | ' + RXM + ' |')
print('|------------------------------------------------------|')
print('')
print('###################################################################')


# In[19]:


print("Thanks for checking out my project assignment! Feel free to look at the source code for further documentation.")
print('')
print('In addition to this program, I also provided a jupyter notebook version in a subfolder if you prefer to read the source that way or test my code more easily.')
print('')
print("Alexander Rusnak - SCIPER #: 309939 - May 2020")
print("EE 559 - Deep Learning, Professor Francois Fleuret")
print('')
print('###################################################################')


# In[ ]:




