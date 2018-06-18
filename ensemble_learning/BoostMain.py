# -*- coding: utf-8 -*-
"""
Created on 2018/6/17 11:07

@author: dmyan
AdaBoost 基分类器使用决策树
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import numpy as np
if __name__ == '__main__':
    train_features = []
    train_labels = []
    test_features = []
    test_labels = []
    T = 10
    dtree = []
    alphas = [0]*T
    with open('./adult_dataset/adult_train_feature.txt', 'r', encoding='utf8') as f:
        line = f.readline()
        while line:
            line = line.strip('\n\r').split(' ')
            train_features.append([float(i) for i in line])
            line = f.readline()

    with open('./adult_dataset/adult_train_label.txt', 'r', encoding='utf8') as f:
        line = f.readline()
        while line:
            line = line.strip('\n\r')
            train_labels.append(int(line))
            line = f.readline()

    with open('./adult_dataset/adult_test_feature.txt', 'r', encoding='utf8') as f:
        line = f.readline()
        while line:
            line = line.strip('\n\r').split(' ')
            test_features.append([float(i) for i in line])
            line = f.readline()

    with open('./adult_dataset/adult_test_label.txt', 'r', encoding='utf8') as f:
        line = f.readline()
        while line:
            line = line.strip('\n\r')
            test_labels.append(int(line))
            line = f.readline()
    train_features = np.array(train_features)
    train_labels = [(-1)**(i+1) for i in train_labels]
    train_labels = np.array(train_labels)
    test_features = np.array(test_features)
    test_labels = [(-1) ** (i + 1) for i in test_labels]
    test_labels = np.array(test_labels)
    d = [1/len(train_labels)]*len(train_labels)
    # clf = DecisionTreeClassifier()
    # clf = clf.fit(train_features, train_labels)
    # y = clf.predict(test_features)
    # y = y.reshape((len(test_labels),1))
    # print(sum(y==test_labels))
    test_labels = np.reshape(test_labels, len(test_labels))
    train_labels = np.reshape(train_labels, len(train_labels))
    for i in range(T):
        clf = DecisionTreeClassifier()
        clf = clf.fit(train_features, train_labels, sample_weight=d)
        dtree.append(clf)
        y = clf.predict(train_features)
        eplison = 1 - clf.score(train_features, train_labels)
        if eplison > 0.5:
            break
        alpha = np.log((1 - eplison)/eplison)/2
        alphas[i] = alpha
        d = d * np.exp(-alpha*y*train_labels)
        d /= sum(d)
    y = [0.0] * len(test_labels)
    # print(dtree[0])
    print(np.shape(alphas))
    for i in range(len(dtree)):
        clf = dtree[i]
        y += alphas[i] * clf.predict(test_features)
    print(sum(np.sign(y) == test_labels)/len(test_labels))


