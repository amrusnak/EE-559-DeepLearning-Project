#!/usr/bin/env python
# coding: utf-8

# In[1]:

print("")
print("Project 1")
print("Alexander Rusnak - SCIPER #: 309939 - May 2020")
print("EE 559 - Deep Learning, Professor Francois Fleuret")
print("")
print("")
print("Below I will train several models, each contextualized in the scope of the project and described as they come up. The final model is the most accurate on the assignment.")
print("")
print("")


# In[2]:



#Importing torch and various torch components
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
from torchvision import datasets

#Importing argparse and os for generating the dataset
import argparse
import os


# In[3]:



#Provided code for generating the dataset

######################################################################

parser = argparse.ArgumentParser(description='DLC prologue file for practical sessions.')

parser.add_argument('--full',
                    action='store_true', default=False,
                    help = 'Use the full set, can take ages (default False)')

parser.add_argument('--tiny',
                    action='store_true', default=False,
                    help = 'Use a very small set for quick checks (default False)')

parser.add_argument('--seed',
                    type = int, default = 0,
                    help = 'Random seed (default 0, < 0 is no seeding)')

parser.add_argument('--cifar',
                    action='store_true', default=False,
                    help = 'Use the CIFAR data-set and not MNIST (default False)')

parser.add_argument('--data_dir',
                    type = str, default = None,
                    help = 'Where are the PyTorch data located (default $PYTORCH_DATA_DIR or \'./data\')')

# Timur's fix
parser.add_argument('-f', '--file',
                    help = 'quick hack for jupyter')

args = parser.parse_args()

if args.seed >= 0:
    torch.manual_seed(args.seed)

######################################################################
# The data

def convert_to_one_hot_labels(input, target):
    tmp = input.new_zeros(target.size(0), target.max() + 1)
    tmp.scatter_(1, target.view(-1, 1), 1.0)
    return tmp

def load_data(cifar = None, one_hot_labels = False, normalize = False, flatten = True):

    if args.data_dir is not None:
        data_dir = args.data_dir
    else:
        data_dir = os.environ.get('PYTORCH_DATA_DIR')
        if data_dir is None:
            data_dir = './data'

    if args.cifar or (cifar is not None and cifar):
        print('* Using CIFAR')
        cifar_train_set = datasets.CIFAR10(data_dir + '/cifar10/', train = True, download = True)
        cifar_test_set = datasets.CIFAR10(data_dir + '/cifar10/', train = False, download = True)

        train_input = torch.from_numpy(cifar_train_set.data)
        train_input = train_input.transpose(3, 1).transpose(2, 3).float()
        train_target = torch.tensor(cifar_train_set.targets, dtype = torch.int64)

        test_input = torch.from_numpy(cifar_test_set.data).float()
        test_input = test_input.transpose(3, 1).transpose(2, 3).float()
        test_target = torch.tensor(cifar_test_set.targets, dtype = torch.int64)

    else:
        print('* Using MNIST')
        mnist_train_set = datasets.MNIST(data_dir + '/mnist/', train = True, download = True)
        mnist_test_set = datasets.MNIST(data_dir + '/mnist/', train = False, download = True)

        train_input = mnist_train_set.data.view(-1, 1, 28, 28).float()
        train_target = mnist_train_set.targets
        test_input = mnist_test_set.data.view(-1, 1, 28, 28).float()
        test_target = mnist_test_set.targets

    if flatten:
        train_input = train_input.clone().reshape(train_input.size(0), -1)
        test_input = test_input.clone().reshape(test_input.size(0), -1)

    if args.full:
        if args.tiny:
            raise ValueError('Cannot have both --full and --tiny')
    else:
        if args.tiny:
            print('** Reduce the data-set to the tiny setup')
            train_input = train_input.narrow(0, 0, 500)
            train_target = train_target.narrow(0, 0, 500)
            test_input = test_input.narrow(0, 0, 100)
            test_target = test_target.narrow(0, 0, 100)
        else:
            print('** Reduce the data-set (use --full for the full thing)')
            train_input = train_input.narrow(0, 0, 1000)
            train_target = train_target.narrow(0, 0, 1000)
            test_input = test_input.narrow(0, 0, 1000)
            test_target = test_target.narrow(0, 0, 1000)

    print('** Use {:d} train and {:d} test samples'.format(train_input.size(0), test_input.size(0)))

    if one_hot_labels:
        train_target = convert_to_one_hot_labels(train_input, train_target)
        test_target = convert_to_one_hot_labels(test_input, test_target)

    if normalize:
        mu, std = train_input.mean(), train_input.std()
        train_input.sub_(mu).div_(std)
        test_input.sub_(mu).div_(std)

    return train_input, train_target, test_input, test_target

######################################################################

def mnist_to_pairs(nb, input, target):
    input = torch.functional.F.avg_pool2d(input, kernel_size = 2)
    a = torch.randperm(input.size(0))
    a = a[:2 * nb].view(nb, 2)
    input = torch.cat((input[a[:, 0]], input[a[:, 1]]), 1)
    classes = target[a]
    target = (classes[:, 0] <= classes[:, 1]).long()
    return input, target, classes

######################################################################

def generate_pair_sets(nb):
    if args.data_dir is not None:
        data_dir = args.data_dir
    else:
        data_dir = os.environ.get('PYTORCH_DATA_DIR')
        if data_dir is None:
            data_dir = './data'

    train_set = datasets.MNIST(data_dir + '/mnist/', train = True, download = True)
    train_input = train_set.data.view(-1, 1, 28, 28).float()
    train_target = train_set.targets

    test_set = datasets.MNIST(data_dir + '/mnist/', train = False, download = True)
    test_input = test_set.data.view(-1, 1, 28, 28).float()
    test_target = test_set.targets

    return mnist_to_pairs(nb, train_input, train_target) +            mnist_to_pairs(nb, test_input, test_target)


# In[4]:



#generate 1000 instances of the training data
train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(1000)


# In[5]:



# My first test is to find a suitable architecture just to classify MNIST images 
# as their correct number. 

# Thus I seperate the data into just the train_input for one image channel, 
# and the train_classes for that same image channel

train1 = torch.utils.data.TensorDataset(train_input[:,:1],train_classes[:,0])
test1 = torch.utils.data.TensorDataset(test_input[:,:1],test_classes[:,0])

# Creating train and test dataloaders for this subset, with a batch size of 8

batchSize = 8

trainLoader1 = torch.utils.data.DataLoader(train1, batch_size = batchSize, shuffle = False)
testLoader1 = torch.utils.data.DataLoader(test1, batch_size = batchSize, shuffle = False)


# In[6]:



# A relatively simple CNN for classifying the MNIST images as their correct digits. 
# I will reuse this architecture later in the document for part of the full model.

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # 3 convolutional layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64,512, kernel_size=3)
        
        # 2 fully connected layers
        self.fc1 = nn.Linear(2*4*64, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # convolutional layer 1 with relu activation
        x = F.relu(self.conv1(x))
        # convolutional layer 2, followed by a 2d pooling layer, with relu activation
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # Dropout layer 1
        x = F.dropout(x, p=0.5, training=self.training)
        # convolutional layer 3, followed by a 2d pooling layer, with relu activation
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        #Dropout layer 2
        x = F.dropout(x, p=0.5, training=self.training)
        # Reshape the tensor for the fully connected layers
        x = x.view(-1,2*4*64 )
        # Fully connected layer 1 with relu
        x = F.relu(self.fc1(x))
        # Dropout 3
        x = F.dropout(x, training=self.training)
        # Fully connected layer 2 with relu
        x = self.fc2(x)
        # Softmax output
        return F.log_softmax(x, dim=1)

# Define first model used to classify the images into the correct digits
mnistClassifierCNN = CNN()


# In[7]:



# Training operation for the first and second model types, takes a model, a loader, and a number of epochs
# Uses adam optimizer and cross entropy loss as the cost function

def fit(model, trainLoader, epochs):
    # Define optimizer as Adam
    optimizer = torch.optim.Adam(model.parameters())
    # Define loss function and cross entropy
    lossFunc = nn.CrossEntropyLoss()
    # Start Training
    model.train()
    # Train for the correct number of epochs
    for epoch in range(epochs):
        # This variable tracks how many correct predictions were made in each epoch
        correct = 0
        # Seperate loader into a batch at each step
        for batch_idx, (X_batch, y_batch) in enumerate(trainLoader):
            
            # Convert batch representations to variable floats for input
            X_V = Variable(X_batch).float()
            y_V = Variable(y_batch)
            # Zero out the gradients 
            optimizer.zero_grad()
            # Get a prediction from the model
            out = model(X_V)
            # Get the loss of the predicted and actual class
            loss = lossFunc(out, y_V)
            # Backpropagate the loss
            loss.backward()
            # optimize the gradients using Adam
            optimizer.step()
            
            # Get the predicition and determine if it is correct
            pred = torch.max(out.data, 1)[1] 
            correct += (pred == y_V).sum()
            
            # Print the loss and accuracy at the end of each epoch
            if batch_idx % 999 == 0:
                print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
                    epoch, batch_idx*len(X_batch), len(trainLoader.dataset), 100.*batch_idx / len(trainLoader), loss.data.item(), float(correct*100) / float(batchSize*(batch_idx+1))))         


# In[8]:



# This function is used to evaluate the models on the test set. 
# It takes 3 arguments, the first two are: the model, and the loader for the test data.
# The last relates to the type of model being passed: full indicates the model is the 
# final version with that takes two inputs (each 14 x 14 image), rather than one input

# All model types are evaluated using the same cross entropy loss.
def evalModel(model, testLoader, full):
    # Counts correct predictions
    correct = 0 
    # Which model type is this? See preceding paragraph
    if(full):
        # Iterates through the loader
            for X_test, y_test in testLoader:
                # Seperates the two images and turns them into float variabless
                V_X_test1 = Variable(X_test[:,:1]).float()
                V_X_test2 = Variable(X_test[:,1:2]).float()
                # Runs model with two inputs, and receives the prediction for the label (out)
                # and the predicted integer value of each MNIST image (outX1 and outX2)
                out, outX1, outX2 = model(V_X_test1, V_X_test2)
                pred = torch.max(out,1)[1]
                # Count if it was correct
                correct += (pred == y_test).sum()
                # Print accuracy data
            print("Test accuracy:  {:.3f}% ".format( float(correct) / (len(testLoader)*batchSize)))
    else:
        # Does the same as above without seperating the input
        for test_X, test_y in testLoader:
            # Turns batch to variable
            test_X = Variable(test_X).float()
            # Gets label prediction
            out = model(test_X)
            pred = torch.max(out,1)[1]
            # Track corrects
            correct += (pred == test_y).sum()
            # Print accuracy
        print("Test accuracy:  {:.3f}% ".format( float(correct) / (len(testLoader)*batchSize)))


# In[9]:

print("")
print("")
print('           -------------------------------------')
# Training the cnnMnist model to predict the class of an MNIST image
print('Model Name: mnistClassifierCNN, Input: One 14x14 MNIST image, Output: The predicted integer class of the MNIST image ')
print('           -------------------------------------')
print('Architecture of the model:')
print(str(mnistClassifierCNN))
print('           -------------------------------------')
fit(mnistClassifierCNN, trainLoader1, 10)
print('           -------------------------------------')
print('')
print('')


# In[10]:



print("Evaluation of mnistClassifierCNN on the test data:")
print('           -------------------------------------')
evalModel(mnistClassifierCNN, testLoader1, False)
print('           -------------------------------------')
print('')
print('')


# In[11]:



# This is the second test model, here I am trying to create a consistent solution to
# predict which integer is larger given the two integer values of the images. 
# This architecture will be partially reused in the full model.

class fullyConnected(nn.Module):
    # On init, this model takes the size of the input and output
    def __init__(self, inputSize, outputSize):
        super(fullyConnected, self).__init__()
        # This model has only 4 linear layers
        self.linear1 = nn.Linear(inputSize,250)
        self.linear2 = nn.Linear(250,250)
        self.linear3 = nn.Linear(250,100)
        self.linear4 = nn.Linear(100,outputSize)
    
    def forward(self,X):
        # Forward pass over the first three linear layers with relu activation
        X = F.relu(self.linear1(X))
        X = F.relu(self.linear2(X))
        X = F.relu(self.linear3(X))
        # Forward pass over the last layer followed by a softmax for output
        X = self.linear4(X)
        return F.log_softmax(X, dim=1)
# initialized this model with an input and output size of two
integerGreaterThanFC = fullyConnected(2, 2)


# In[12]:



# Create two new dataloaders, this time taking the classes as input data, and the targets 
# as output data.
train2 = torch.utils.data.TensorDataset(train_classes, train_target)
test2 = torch.utils.data.TensorDataset(test_classes, test_target)

testLoader2 = torch.utils.data.DataLoader(test2, batch_size = batchSize, shuffle = False)
trainLoader2 = torch.utils.data.DataLoader(train2, batch_size = batchSize, shuffle = False)


# In[13]:



print('Model Name: integerGreaterThanFC, Input: Integer value of the two images (train_classes), Output: Predicts which is larger (train_labels)')
print('           -------------------------------------')
print('Architecture of the model:')
print(str(integerGreaterThanFC))
print('           -------------------------------------')
# Training the model
fit(integerGreaterThanFC, trainLoader2, 10)
print('           -------------------------------------')
print('')
print('')


# In[14]:



print("Evaluation of integerGreaterThanFC on the test data:")
print('           -------------------------------------')
evalModel(integerGreaterThanFC, testLoader2, False)
print('           -------------------------------------')
print('')
print('')


# In[15]:



# Create a third train and test loader, with the train loader taking all three datasets
train3 = torch.utils.data.TensorDataset(train_input, train_target, train_classes)
test3 = torch.utils.data.TensorDataset(test_input, test_target)

trainLoader3 = torch.utils.data.DataLoader(train3, batch_size = batchSize, shuffle = False)
testLoader3 = torch.utils.data.DataLoader(test3, batch_size = batchSize, shuffle = False)


# In[16]:



# This is the full model combining the two previous components, as outlined in the project 
# brief, it uses shared weights and an auxiliary loss function for correctly classifying 
# the digits before picking which is larger
class fullModel(nn.Module):
    def __init__(self):
        super(fullModel, self).__init__()
        # The model starts with 3 convolutional layers, in application both images will 
        # be passed through them seperately, but sharing the weights
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64,512, kernel_size=3)
        # Two fully connected layers to downsize the convolutional layers to a digit
        # prediction
        self.fc1 = nn.Linear(2*4*64, 256)
        self.fc2 = nn.Linear(256, 10)
        # Three fully connected layers to map the digits to the target, i.e. which is 
        # larger
        self.linear1 = nn.Linear(20,100)
        self.linear2 = nn.Linear(100,50)
        self.linear3 = nn.Linear(50,2)

    def forward(self, x1, x2):
        # Forward accepts two arguments, x1 and x2, each representing one of the two images
        # Pass both images through the first conv layer seperately with relu
        x1 = F.relu(self.conv1(x1))
        x2 = F.relu(self.conv1(x2))
        # Pass both images through the second conv layer seperately with max pool and relu
        x1 = F.relu(F.max_pool2d(self.conv2(x1), 2))
        x2 = F.relu(F.max_pool2d(self.conv2(x2), 2))
        # Dropout 1
        x1 = F.dropout(x1, p=0.5, training=self.training)
        x2 = F.dropout(x2, p=0.5, training=self.training)
        # Both images through third conv layer, with max pool and relu
        x1 = F.relu(F.max_pool2d(self.conv3(x1),2))
        x2 = F.relu(F.max_pool2d(self.conv3(x2),2))
        # Dropout 2
        x1 = F.dropout(x1, p=0.5, training=self.training)
        x2 = F.dropout(x2, p=0.5, training=self.training)
        # Reshape tensor for fc layers
        x1 = x1.view(-1,2*4*64 )
        x2 = x2.view(-1,2*4*64 )
        # First fully connected layer with relu
        x1 = F.relu(self.fc1(x1))
        x2 = F.relu(self.fc1(x2))
        # Dropout 3
        x1 = F.dropout(x1, training=self.training)
        x2 = F.dropout(x2, training=self.training)
        # 2nd fully connected layer with relu
        x1 = self.fc2(x1)
        x2 = self.fc2(x2)
        # Softmax the prediction of each image digit
        x_f_1 = F.log_softmax(x1, dim=1)
        x_f_2 = F.log_softmax(x2, dim=1)
        # Combine the digit predictions into a single input tensor
        z = torch.cat((x_f_1, x_f_2), 1)
        # Third fully connected layer, this one for the digits, with relu
        z = F.relu(self.linear1(z))
        # Dropout 4
        z = F.dropout(z, training=self.training)
        # Fourth fully connected layer, with relu
        z = F.relu(self.linear2(z))
        # Fifth fully connected layer, without relu
        z = self.linear3(z)
        # Returns the softmax of the final output corresponing to the target, 
        # and x_f_1, x_f_2, corresponding to the digit prediction for each image
        # which will be used for the auxiliary loss
        return F.log_softmax(z, dim=1), x_f_1, x_f_2
# Initialize two models of this type, one to be trained without the auxiliary loss and one with
sharedWeights = fullModel()
sharedWeightsWithAuxLoss = fullModel()


# In[17]:



# I've defined here another fucntion for training the model, this one specifically for the
# full model. I could have combined both fit functions into one with if statements, but I 
# think seperating them is more clear for reading through the code
def fitFullModel(model, trainLoader, epochs, aux):
    # Same optimizer and loss as before (adam and cross entropy)
    optimizer = torch.optim.Adam(model.parameters())
    lossFunc = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        # Tracks correct answers
        correct = 0
        # This trainLoader outputs all three datasets at each batch step
        for batch_idx, (X_batch, y_batch, z_batch) in enumerate(trainLoader):
            # Turn inout images into float variables
            X_V1 = Variable(X_batch[:,:1]).float()
            X_V2 = Variable(X_batch[:,1:2]).float()
            y_V = Variable(y_batch)
            # Zero the gradients
            optimizer.zero_grad()
            # Run the model and accept prediction out for the target (which is larger), 
            # and the predicted digit classes of each image (outX1, outX2)
            out, outX1, outX2 = model(X_V1, X_V2)
            # Calculate the loss for the final output in relation ot the target
            loss1 = lossFunc(out, y_V)
            # If using auxiliary loss
            if(aux):
                # Calculate the loss for the digit classification for each image
                loss2 = lossFunc(outX1, z_batch[:,0])
                loss3 = lossFunc(outX2, z_batch[:,1])
                # Combine the loss values into one loss value. First combine the 
                # loss for the images, then combine that value with the target loss. 
                # The target loss is wiehgted at 50%, each image at 25%
                lossCharacter = (loss2 + loss3)/2
                fullCombinedLoss = (loss1 + lossCharacter)/2
            # If not using aux, just set the loss to the target loss
            else:
                fullCombinedLoss = loss1
            # Backpropagate the loss
            fullCombinedLoss.backward()
            # Optimize the gradients
            optimizer.step()
            # Get prediction and track is it is correct
            pred = torch.max(out.data, 1)[1] 
            correct += (pred == y_V).sum()
            # Print results
            if batch_idx % 1000 == 0:
                print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
                    epoch, batch_idx*len(X_batch), len(trainLoader.dataset), 100.*batch_idx / len(trainLoader), fullCombinedLoss.data.item(), float(correct*100) / float(batchSize*(batch_idx+1))))


# In[18]:



print('Model Name: sharedWeights, Input: The 14x14 MNIST image data for both digit images, Output: Predicts which is larger (train_labels)')
print('           -------------------------------------')
print('Architecture of the model:')
print(str(sharedWeights))
print('           -------------------------------------')
print('Key Stipulation: This model passes each image through the initial convolutional layers seperately, but the layers only have one set of weights.')
print('           -------------------------------------')
print('Key Stipulation: This model is not being trained with the auxiliary loss function.')
print('           -------------------------------------')
# Training the model
fitFullModel(sharedWeights, trainLoader3, 10, False)
print('           -------------------------------------')
print('')
print('')


# In[19]:



print("Evaluation of sharedWeights on the test data:")
print('           -------------------------------------')
evalModel(sharedWeights, testLoader3, True)
print('           -------------------------------------')
print('')
print('')


# In[20]:



print('Model Name: sharedWeightsWithAuxLoss, Input: The 14x14 MNIST image data for both digit images, Output: Predicts which is larger (train_labels)')
print('           -------------------------------------')
print('Architecture of the model:')
print(str(sharedWeightsWithAuxLoss))
print('           -------------------------------------')
print('Key Stipulation: This model has the same architecture as above, the only difference is the loss.')
print('           -------------------------------------')
print('Key Stipulation: This model is using the auxiliary loss function. This function calculates the loss of the predicted digits in the middle of the model in addition to the final target loss. The target loss is weighted at 50%, and the loss of each image at 25%.')
print('           -------------------------------------')
# Training the model
fitFullModel(sharedWeightsWithAuxLoss, trainLoader3, 10, True)
print('           -------------------------------------')
print('')
print('')


# In[21]:



print("Evaluation of sharedWeightsWithAuxLoss on the test data:")
print('           -------------------------------------')
evalModel(sharedWeightsWithAuxLoss, testLoader3, True)
print('           -------------------------------------')
print('^ As you can see from the above, using the auxiliary loss helps the model to be more accurate. ^')
print('           -------------------------------------')
print('')
print('')


# In[22]:



print("Thanks for checking out my project assignment! Feel free to look at the source code.")
print("Alexander Rusnak - SCIPER #: 309939 - May 2020")
print("EE 559 - Deep Learning, Professor Francois Fleuret")
print('')
print('')


# In[ ]:




