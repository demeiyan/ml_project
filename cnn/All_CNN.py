# -*- coding: utf-8 -*-
"""
Created on 18-5-4 下午5:21

@author: dmyan
"""
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

def dataLoader():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                             shuffle=False, num_workers=2)
    return trainloader, testloader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.5)
        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=0)
        self.conv8 = nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0)
        self.conv9 = nn.Conv2d(192, 10, kernel_size=1, stride=1, padding=0)
        self.pool = nn.AvgPool2d(6)

    def forward(self, x):
        x = self.drop1(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(self.drop2(x)))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(self.drop2(x)))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = self.pool(x)
        return x


if __name__ == '__main__':
    # learning_rate weight_decay
    # 0.0001    0.0005
    # 0.0003    0.00008
    # 0.0004    0.00007 75.5%
    # 0.0005    0.001   79%
    # 0.00005   0.00025 69.450%	1400 epoch 
    # 0.00008	0.00015 72.61%	300 epoch
    # 0.0005	0.00055
    # 0.0005	0.00055 75.81%	300 epoch
    # 0.001     0.00015 82.9%   300 epoch
    # 0.001     0.0001  81.8%   3000 epoch
    batch_count = 0
    x_count = []
    y_loss = []
    max_acc = -1
    max_epoch = -1
    batch_size = 256
    learning_rate = 0.001
    decay = 0.00015	
    net = Net()
    if torch.cuda.is_available():
        net = net.cuda()
    running_loss = 0.0
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    criterion = nn.CrossEntropyLoss()
    s = [200, 250, 300]
    trainloader, testloader = dataLoader()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=decay)
    for epoch in range(500):  # loop over the dataset multiple times

        if epoch == s[0] or epoch == s[1] or epoch == s[2]:
            learning_rate = learning_rate*0.1
            optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=decay)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = Variable(inputs).cuda()
                labels = Variable(labels).cuda()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net.forward(inputs)
            loss = criterion(outputs[:, :, 0, 0], labels)
            loss.backward()
            optimizer.step()

            # print statistics
            #x_count.append(batch_count)
            #y_loss.append(loss.item())
            #batch_count += 1
            running_loss += loss.item()
            if i % 195 == 194:  # print every 195 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 195))
                running_loss = 0.0
        if epoch % 50 == 49:    # test every 50 epoch
            class_correct = list(0. for i in range(10))
            class_total = list(0. for i in range(10))
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    images = images.cuda()
                    labels = labels.cuda()
                    outputs = net.forward(images)
                    outputs = torch.argmax(outputs[:, :, 0, 0], dim=1)
                    c = (outputs == labels)
                    for i in range(list(labels.shape)[0]):
                        label = labels[i]
                        class_correct[label] += c[i].item()
                        class_total[label] += 1

            for i in range(10):
                print('Accuracy of %5s : %2d %%' % (
                    classes[i], 100 * class_correct[i] / class_total[i]))
            acc = sum(class_correct)/sum(class_total)
            print('Total Accuracy is : %.3f%%' % (acc*100))
            if acc > max_acc:
                max_acc = acc
                max_epoch = epoch
    print('Finished Training')
    # plt.figure()
    # plt.plot(x_count, y_loss)
    # plt.xlabel('batch_count')
    # plt.ylabel('loss')
    # plt.savefig('./loss.png')
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    # test
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = net.forward(images)
            outputs = torch.argmax(outputs[:, :, 0, 0], dim=1)
            c = (outputs == labels)
            for i in range(list(labels.shape)[0]):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
    print('Total Accuracy is : %.3f%%' % (sum(class_correct) * 100 / sum(class_total)))
    print("max_epoch : %d max_acc : %.3f learning_rate : %.5f decay : %.5f" % ( max_epoch, max_acc, learning_rate, decay))

