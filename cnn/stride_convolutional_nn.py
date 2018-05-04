# -*- coding: utf-8 -*-
"""
Created on 18-5-4 下午5:21

@author: dmyan
"""
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


def dataLoader():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)
    return trainloader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.drop1 = nn.Dropout()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=0)
        self.conv8 = nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0)
        self.conv9 = nn.Conv2d(192, 10, kernel_size=1, stride=1, padding=0)
        self.pool = nn.AvgPool2d(kernel_size=0, stride=1, padding=0)

    def forward(self, x):
        x = self.drop1(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.drop1(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.drop1(self.conv6(x)))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = self.pool(x)
        return x


if __name__ == '__main__':
    net = Net()
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    trainloader = dataLoader()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    i = 0
    for data in trainloader:
        inputs, labels = data
        print(inputs, labels)
        break

    print('Finished Training')