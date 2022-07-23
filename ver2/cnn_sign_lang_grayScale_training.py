# -*- coding: utf-8 -*-
"""
Created on Sat May 28 08:37:56 2022

@author: H
"""

"""
Created on Thu May 26 10:19:32 2022

@author: H
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import warnings

warnings.filterwarnings('ignore')

# parameters
batch_size = 32
epochs =10
learning_rate = 0.001
#classes = ()
device ="cuda:0" #torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
torch.manual_seed(32)

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(0.5,0.5),
    transforms.Grayscale(num_output_channels=1)
])
test_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(0.5,0.5),
    transforms.Grayscale(num_output_channels=1)
])

#data_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1)])

dataset_path = "C:/Users/H/OneDrive/سطح المكتب/SignLanguage/Datasets/1/ArASL_Database_54K_Final"

dataset = datasets.ImageFolder(dataset_path)
#breakpoint()
train_indices, test_indices = train_test_split(list(range(len(dataset.targets))), test_size=0.2,
                                               stratify=dataset.targets)

train_dataset = torch.utils.data.Subset(dataset, train_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)
#print(train_dataset[0])
#breakpoint()
train_dataset.dataset.transform = train_transform
test_dataset.dataset.transform = test_transform

trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(8 * 8 * 64, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 512),

            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, 32),

         
        )

    def forward(self, x):
        # conv layers
        #breakpoint()
        x = self.conv_layer(x)
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = self.fc_layer(x)
        return x


net = CNN().to(device)



def train_model(net, trainloader, epochs=60
                , learning_rate=0.001, verbose=1, print_every=1):
    loss_list = []
    accuracy_list = []

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(epochs):
        epoch_accuracy_list = []

        for i, (inputs, target) in enumerate(trainloader):
            #breakpoint()
            inputs = inputs.to(device)
            target = target.to(device)

            outputs = net(inputs)
            prediction = torch.max(outputs, 1)[1]
            epoch_accuracy_list.append(accuracy_score(target.cpu(), prediction.cpu()))
            
            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_list.append(loss.item())
        accuracy = np.array(epoch_accuracy_list).mean()
        print(accuracy)
        accuracy_list.append(accuracy)
       

        if (verbose):
            if ((epoch + 1) % print_every == 0) :
                print('epoch {}/{}: loss {:.3f} accuracy {:.3f}'.format(epoch + 1, epochs, loss.item(), accuracy))
    return loss_list, accuracy_list


loss, accuracy = train_model(net, trainloader=trainloader,learning_rate=learning_rate,epochs=epochs, verbose=1, print_every=1)
print(loss,accuracy)
torch.save(net.state_dict(), "project_signLanguage.pt")
plt.figure()
plt.plot(loss)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("training loss")

plt.figure()
plt.plot(accuracy)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.title("training accuracy")

plt.show()

net.eval()
target_list = []
prediction_list = []
with torch.no_grad():
    for i, (inputs, target) in enumerate(testloader):
        inputs = inputs.to(device)
        outputs = net(inputs).cpu()
        prediction = torch.max(outputs, 1)[1]

        target_list.append(target)
        prediction_list.append(prediction)

target = np.hstack(target_list)
prediction = np.hstack(prediction_list)
print(classification_report(np.hstack(target_list), np.hstack(prediction_list))) #target_names=classes))
#print(confusion_matrix(target, prediction))
