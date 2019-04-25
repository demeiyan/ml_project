# -*- coding: utf-8 -*-
"""
Created on 2018/6/17 11:07

@author: dmyan
"""
from BoostMain import load_features_labels
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def Random_Forest(T, x_train, y_train, x_test, y_test):

    y = [0.0] * len(y_test)
    for i in range(T):
        features, labels = resample(x_train, y_train, replace=True)
        clf = DecisionTreeClassifier(criterion="gini", max_features="log2")
        clf.fit(features, labels)
        y += clf.predict(x_test)
    y = np.sign(y)
    return y

def cross_validation(T, train_features, train_labels):

    train_features = np.array(train_features)
    train_labels = [(-1)**(i+1) for i in train_labels]
    train_labels = np.array(train_labels)

    mean_auc = 0.0

    for j in range(5):
        x_train, x_test, y_train, y_test = train_test_split(train_features, train_labels, test_size=0.2,
                                                            stratify=train_labels)
        y = Random_Forest(T, x_train, y_train, x_test, y_test)
        fpr, tpr, thresholds = roc_curve(y_test, y)
        mean_auc += auc(fpr, tpr)
    return mean_auc / 5

if __name__ == '__main__':
    T = 50
    dtree = []
    train_features, train_labels = load_features_labels('./adult_dataset/adult_train_feature.txt', './adult_dataset/adult_train_label.txt')
    # test_features, test_labels = load_features_labels('./adult_dataset/adult_test_feature.txt', './adult_dataset/adult_test_label.txt')

    x = []
    aucs = []
    for i in range(1, 11):
        x.append(i * 5)
        aucs.append(cross_validation(i * 5, train_features, train_labels))

    plt.plot(x, aucs)
    plt.title('Random Forest performance')
    plt.xlabel("The number of base learner")
    plt.ylabel('The AUC of 5-flod cross validation')
    plt.subplots_adjust(left=0.15, bottom=0.15)
    #plt.show()
    plt.savefig('Random_Forest.png')

