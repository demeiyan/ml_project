# -*- coding: utf-8 -*-
"""
Created on 2018/6/17 11:07

@author: dmyan
AdaBoost 基分类器使用决策树
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def load_features_labels(features_path, label_path):
    features = []
    labels = []
    with open(features_path, 'r', encoding="utf8") as features_files,\
        open(label_path, 'r', encoding="utf8") as labels_files:
        feas = features_files.readlines()
        labs = labels_files.readlines()
        for f, l in zip(feas, labs):
            features.append([float(i) for i in f.strip('\n\r').split(' ')])
            labels.append(int(l.strip("\n\r")))
    return features, labels


def AdaBoost(T, x_train, y_train, x_test, y_test):

    d = [1 / len(y_train)] * len(y_train)
    dtree = []
    alphas = [0] * T
    for i in range(T):
        clf = DecisionTreeClassifier(criterion="gini")
        clf.fit(x_train, y_train, sample_weight=d)
        dtree.append(clf)
        y = clf.predict(x_train)
        eplison = 1 - clf.score(x_train, y_train, sample_weight=d)
        if eplison > 0.5 or eplison <= 0.0:
            break
        alpha = np.log((1 - eplison) / eplison) / 2
        alphas[i] = alpha

        d = d * np.exp(-alpha * y * y_train)
        d /= sum(d)

    y = [0.0] * len(y_test)
    for i in range(len(dtree)):
        clf = dtree[i]
        if clf:
            y += alphas[i] * clf.predict(x_test)
    y = np.sign(y)
    return y

def cross_validation(T, train_features, train_labels):

    #test_features, test_labels = load_features_labels('./adult_dataset/adult_test_feature.txt', './adult_dataset/adult_test_label.txt')

    train_features = np.array(train_features)
    train_labels = [(-1)**(i+1) for i in train_labels]
    train_labels = np.array(train_labels)
    #test_features = np.array(test_features)
    #test_labels = [(-1) ** (i + 1) for i in test_labels]
    #test_labels = np.array(test_labels)

    mean_auc = 0.0
    for j in range(5):
        x_train, x_test, y_train, y_test = train_test_split(train_features, train_labels, test_size=0.2,
                                                            stratify=train_labels)

        y = AdaBoost(T, x_train, y_train, x_test, y_test)
        fpr, tpr, thresholds = roc_curve(y_test, y)
        mean_auc += auc(fpr, tpr)
    return mean_auc/5


if __name__ == '__main__':
    train_features, train_labels = load_features_labels('./adult_dataset/adult_train_feature.txt',
                                                        './adult_dataset/adult_train_label.txt')
    x = []
    aucs = []
    for i in range(1, 11):
        x.append(i * 5)
        aucs.append(cross_validation(i * 5, train_features, train_labels))

    plt.plot(x, aucs)
    plt.title('AdaBoost performance')
    plt.xlabel("The number of base learner")
    plt.ylabel('The AUC of 5-flod cross validation')
    #plt.show()
    plt.subplots_adjust(left=0.15, bottom=0.15)
    plt.savefig('AdaBoost.png')

