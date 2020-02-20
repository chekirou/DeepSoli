import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fc
import torch.nn.modules.normalization as nm
import torch.optim as optim

class CNN(nn.Module):

    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,32, kernel_size=3),
            nn.ReLU(inplace=True),
            nm.LocalResponseNorm(size=3),
            nn.MaxPool2d(kernel_size=2,stride=3)
        )
        self.conv2= nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3,stride=1,padding=0),
            nn.ReLU(inplace=True),
            nm.LocalResponseNorm(size=3),
            nn.MaxPool2d(kernel_size=2,stride=3)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3,stride=1,padding=0),
            nn.ReLU(inplace=True),
            nm.LocalResponseNorm(size=3),
            nn.MaxPool2d(kernel_size=2,stride=3)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.9, inplace=True),
            nn.Linear(7*7*128,512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8,inplace=True),
            nn.Linear(512,512),
            nn.ReLU(inplace=True),
            nn.Linear(512,11),
            nn.Softmax(dim=1),
            )
    def forward(self, img):
        output = self.conv1(img)
        output = self.conv2(output)
        output = self.conv3(output)
        output = output.view(output.shape[0],-1)
        return self.classifier(output)
    def train():
      self.train()
      for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
          train_losses.append(loss.item())
          train_counter.append(
            (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
          torch.save(network.state_dict(), '/results/model.pth')
          torch.save(optimizer.state_dict(), '/results/optimizer.pth')
