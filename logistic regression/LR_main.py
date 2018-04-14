# -*- coding: utf-8 -*-
"""
Created on 2018/4/14 20:39

@author: dmyan
"""
import numpy as np


def smote():
    pass


def convertdata(feature, label):
    x = []
    y = []
    with open('./assign2_dataset/'+feature,'r') as f:
        for line in f.readlines():
            if len(line.strip()) > 0:
                x.append([float(x) for x in line.strip().split(' ')])
    x = np.array(x)
    with open('./assign2_dataset/'+label, 'r') as f:
        for line in f.readlines():
            y.append([float(x) for x in line.strip().split(' ')])
    y = np.array(y)
    print(np.shape(x),np.shape(y))
    return x, y


if __name__ == '__main__':
    x_train, label_train = convertdata('page_blocks_train_feature.txt', 'page_blocks_train_label.txt')
    x_test, label_test = convertdata('page_blocks_test_feature.txt', 'page_blocks_test_label.txt')
    print(x_train, label_train, x_test, label_test)