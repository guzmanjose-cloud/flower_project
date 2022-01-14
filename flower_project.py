import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt

if torch.cuda.is_available():
  device = torch.device("cuda:0")
  print('GPU Enabled, training on GPU...')
else:
  device = torch.device("cpu")
  print('no GPU, training on CPU...')

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

data_valid = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])]) 

# TODO: Load the datasets with ImageFolder
image_datasets = datasets.ImageFolder(train_dir, transform=data_transforms)
validation_data = datasets.ImageFolder(valid_dir, transform = data_valid)

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(validation_data, batch_size = 64, shuffle = True)

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# TODO: Build and train your network
vgg = models.vgg16(pretrained=True)





for param in vgg.parameters():
    param.requires_grad = False

from collections import OrderedDict


classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 512, bias=True)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(512, 102, bias=True)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
vgg.classifier = classifier

criterion = nn.NLLLoss()
optimizer = optim.Adam(vgg.classifier.parameters(), lr=0.001)

epochs = 10

for e in range(epochs):
 
    running_loss = 0

    step = 0
    
    train_losses, test_losses = [], []
    for images, labels in dataloaders:
        images, labels = images.to(device), labels.to(device)
        vgg.to(device)
 
        step += 1
 
        # TODO: Training pass
 
        optimizer.zero_grad()
 
        output = vgg.forward(images)

        loss = criterion(output, labels)

        

        loss.backward()

        optimizer.step()

        

        running_loss += loss.item()

        #print(f"Training loss: {running_loss/len(dataloaders)}")

        print(f"Step: {step} Training loss: {running_loss/step}")
    else: 
        test_loss = 0
        accuracy = 0
        
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            for images, labels in validloader:
                images, labels = images.to(device), labels.to(device)
                vgg.to(device)
                log_ps = vgg(images)
                test_loss += criterion(log_ps, labels)
                
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
                
        train_losses.append(running_loss/len(dataloaders))
        test_losses.append(test_loss/len(validloader))

        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss/len(dataloaders)),
              "Test Loss: {:.3f}.. ".format(test_loss/len(validloader)),
              "Test Accuracy: {:.3f}".format(accuracy/len(validloader)))

#Validation on the test set
correct = 0
total = 0
#Itterare over the test set
with torch.no_grad():
    for data in validloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = vgg(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))