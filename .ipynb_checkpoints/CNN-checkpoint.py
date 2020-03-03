import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fc
import torch.nn.modules.normalization as nm
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 3)
        self.pool = nn.MaxPool2d(2, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(7 * 7 * 128, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 11)

    def forward(self, x):
        x = self.pool(fc.relu(self.conv1(x)))
        x = self.pool(fc.relu(self.conv2(x)))
        x = self.pool(fc.relu(self.conv3(x)))
        x = x.view( 7 * 7 * 128,-1)
        x = fc.relu(self.fc1(x))
        x = fc.relu(self.fc2(x))
        x = self.fc3(x)
        return fc.softmax(x,dim=1)
    
class ShallowCNN(nn.Module):

    def __init__(self):
        super(ShallowCNN,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4,32, kernel_size=3),
            nn.ReLU(),
            nm.LocalResponseNorm(size=3),
            nn.MaxPool2d(kernel_size=2,stride=3)
        )
        self.conv2= nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3,stride=1,padding=0),
            nn.ReLU(),
            nm.LocalResponseNorm(size=3),
            nn.MaxPool2d(kernel_size=2,stride=3)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3,stride=1,padding=0),
            nn.ReLU(),
            nm.LocalResponseNorm(size=3),
            nn.MaxPool2d(kernel_size=2,stride=3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(7*7*128,512),
            nn.ReLU(),
            nn.Dropout(0.9),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(512,11),
            nn.Softmax(dim=1),
            )
    def forward(self, img):
        output = self.conv1(img)
        output = self.conv2(output)
        output = self.conv3(output)
        output = output.view(output.shape[0],-1)
        output = self.classifier(output)
        return output

    
    
class DeepCNN(nn.Module):

    def __init__(self):
        super(DeepCNN,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 96, kernel_size=7),
            nn.ReLU(inplace=True),
            nm.LocalResponseNorm(size=3),
            nn.MaxPool2d(kernel_size=3,stride=7)
        )
        self.conv2= nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5,stride=1,padding=0),
            nn.ReLU(inplace=True),
            nm.LocalResponseNorm(size=3),
            nn.MaxPool2d(kernel_size=2,stride=5)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3,stride=1,padding=0),
            nn.ReLU(inplace=True),
            nm.LocalResponseNorm(size=3),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3,stride=1,padding=0),
            nn.ReLU(inplace=True),
            nm.LocalResponseNorm(size=3),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3,stride=1,padding=0),
            nn.ReLU(inplace=True),
            nm.LocalResponseNorm(size=3),
            nn.MaxPool2d(kernel_size=2,stride=3)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.9, inplace=True),
            nn.Linear(21*21*512,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8,inplace=True),
            nn.Linear(4096,2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048,11),
            nn.Softmax(dim=1),
            )
    def forward(self, img):
        output = self.conv1(img)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = output.view(output.shape[0],-1)
        return self.classifier(output)
       