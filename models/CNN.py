import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional

class CNN(nn.Module):

    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.localResponseNorm(size=)
            nn.MaxPool2d(kernel_size=2,stride=)
        )
        self.conv2= nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3,stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.localResponseNorm(size=)
            nn.MaxPool2d(kernel_size=2,stride=1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3,stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.localResponseNorm(size=)
            nn.MaxPool2d(kernel_size=2,stride=1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(,512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8,inplace=True),
            nn.Linear(512,512),
            nn.ReLU(inplace=True),
            nn.Softmax(),
            )
    def forward(self, img):
        output = self.conv1(img)
        output = self.conv2(output)
        output = self.conv3(output)
        output = output.view(output.shape[0],-1)
        return self.classifier(output)

model = CNN()
