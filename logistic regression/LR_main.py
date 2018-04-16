# -*- coding: utf-8 -*-
"""
Created on 2018/4/14 20:39

@author: dmyan
使用二分类来实现多分类问题,OvR(一对其余)
"""
import numpy as np


def trainValidSplit(feature, label, valid_size=0.1):
    train_indices = []
    valid_indices = []
    for i in range(5):
        index = np.where(label == (i+1))[0]
        np.random.shuffle(index)
        rand = np.random.randint(index.shape[0], size=int(index.shape[0]*valid_size))
        valid_indices.append(index[rand].tolist())
        index = np.delete(index, rand)
        train_indices.append(index.tolist())
    train_indices = [item for sublist in train_indices for item in sublist]
    valid_indices = [item for sublist in valid_indices for item in sublist]
    return feature[train_indices], label[train_indices], feature[valid_indices], label[valid_indices]


def smote():
    pass


def sigmoid(belta, x):
    try:
        return np.exp(np.dot(belta, x))/(1+np.exp(np.dot(belta, x)))
    except Exception as insit:
        print(insit)
        print(belta, x, np.dot(belta, x))


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


def binaryclassification(x, y, one, belta, learn_rate):
    y = np.array([int(i == one) for i in y])
    der = sigmoid(belta, x.T) - y # sigmoid(belta, x.T) - y
    der = np.reshape(der, (y.shape[0], 1))
    der = np.sum(x*der, axis=0)
    belta = belta - learn_rate*der
    return belta


if __name__ == '__main__':
    min_batch = 64
    learn_rate = 0.025
    validation_size = 0.1
    x_train, label_train = convertdata('page_blocks_train_feature.txt', 'page_blocks_train_label.txt')
    x_test, label_test = convertdata('page_blocks_test_feature.txt', 'page_blocks_test_label.txt')
    # 归一化,对特征的每一个维度计算x = (x-mean)/std 归一化到正态分布
    std = x_train.std(axis=0)
    mean = x_train.mean(axis=0)
    x_train = (x_train - mean)/std
    x_test = (x_test - mean)/std
    x_train = np.concatenate((x_train, np.ones((1, x_train.shape[0])).T), axis=1)
    x_test = np.concatenate((x_test, np.ones((1, x_test.shape[0])).T), axis=1)
    label_train = np.array([int(x) for x in label_train])
    label_test = np.array([int(x) for x in label_test])
    # x_train, label_train, x_valid, label_valid =
    x_train, label_train, x_valid, label_valid = trainValidSplit(x_train, label_train, 0.1)

    datasize = label_train.shape[0]
    belta = np.zeros((5, 11))


    # belta[0] = binaryclassification(x_train[0 * min_batch:(0 + 1) * min_batch],label_train[0 * min_batch:(0 + 1) * min_batch], 5, belta[0], learn_rate)
    # for epoch in range(100):
    #     print(belta[0])
    #     for i in range(int(datasize/min_batch)):
    #         belta[0] = binaryclassification(x_train[i*min_batch:(i+1)*min_batch], label_train[i*min_batch:(i+1)*min_batch], 5, belta[0], learn_rate)
    for epoch in range(100):
        for i in range(5):
            for j in range(int(datasize/min_batch)):
                belta[i] = binaryclassification(x_train[j*min_batch:(j+1)*min_batch], label_train[j*min_batch:(j+1)*min_batch], i+1, belta[i], learn_rate)
        # print(belta)
        predict = sigmoid(belta, x_test.T)
        predict = np.argmax(predict, axis=0) + 1
        # print(predict)
        print('epoch %d %.2f%%' % (epoch, (predict == label_test).sum()/label_test.shape[0]*100))
