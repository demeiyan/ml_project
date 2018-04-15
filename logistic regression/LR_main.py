# -*- coding: utf-8 -*-
"""
Created on 2018/4/14 20:39

@author: dmyan
使用二分类来实现多分类问题,OvR(一对其余)
"""
import numpy as np


def smote():
    pass


def sigmoid(x, belta):
    return np.exp(np.dot(x, belta))/(1+np.exp(np.dot(x, belta)))


def convertdata(feature, label):
    x = []
    y = []
    with open('./assign2_dataset/'+feature, 'r') as f:
        for line in f.readlines():
            if len(line.strip()) > 0:
                x.append([float(x) for x in line.strip().split(' ')])
    x = np.array(x)
    with open('./assign2_dataset/'+label, 'r') as f:
        for line in f.readlines():
            y.append([float(x) for x in line.strip().split(' ')])
    y = np.array(y)
    return x, y


def binaryclassification(x, y, one, belta,learn_rate):
    y = np.array([i & one for i in y])
    der = y - sigmoid(x, belta)
    der = np.reshape(der, (y.shape[0], 1))
    der = np.sum(x*der, axis=0)
    belta = belta - learn_rate*der
    return belta


if __name__ == '__main__':
    min_batch = 15
    learn_rate = 0.001
    x_train, label_train = convertdata('page_blocks_train_feature.txt', 'page_blocks_train_label.txt')
    x_test, label_test = convertdata('page_blocks_test_feature.txt', 'page_blocks_test_label.txt')
    # 归一化,对特征的每一个维度计算x = (x-mean)/std归一化到正态分布
    std = x_train.std(axis=0)
    mean = x_train.mean(axis=0)
    x_train = (x_train - mean)/std
    x_test = (x_test - mean)/std
    # print(mean, x_train)
    x_train = np.concatenate((x_train, np.ones((1, x_train.shape[0])).T), axis=1)
    x_test = np.concatenate((x_test, np.ones((1, x_test.shape[0])).T), axis=1)
    label_train = np.array([int(x) for x in label_train])
    label_test = np.array([int(x) for x in label_test])
    datasize = label_train.shape[0]
    belta = np.zeros((5, 11))
    #binaryclassification(x_train[0:64], label_train[0:64], 1, belta[1], learn_rate)
    # for epoch in range(10)
    for i in range(5):
        for j in range(int(datasize/min_batch)):
            belta[i] = binaryclassification(x_train[j*min_batch:(j+1)*min_batch], label_train[j*min_batch:(j+1)*min_batch], i+1, belta[i], learn_rate)
    print(belta)

    # print(np.shape(x_train[0:min_batch]))
    # print(np.shape(label_train))