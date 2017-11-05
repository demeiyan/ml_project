# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 24:00:00 2017

@author: Zhiyu
"""
# import your module here
import numpy as np
import time
import pickle as pkl
from threading import Thread
import functools
# (global) variable definition here

TRAINING_TIME_LIMIT = 60 * 10
EPS = np.finfo(float).eps
# class definition here
# function definition here
def timeout(timeout):
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [Exception('function [%s] timeout [%s seconds] exceeded!' % (func.__name__, timeout))]

            def newFunc():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e

            t = Thread(target=newFunc)
            t.daemon = True
            try:
                t.start()
                t.join(timeout)
            except Exception as je:
                print('error starting thread')
                raise je
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret

        return wrapper

    return deco


@timeout(TRAINING_TIME_LIMIT)
def train(traindata):
    """
    X: data matrix, (n x d)
    y: scalar labels, (n)
    """
    max_iter = 100
    learning_rate = 0.005
    X = traindata[0]
    labels = traindata[1]
    n, d = X.shape
    global A
    A =  np.zeros((d, d))
    np.fill_diagonal(A, 1)
    #psum =0.0
    for it in range(10):
        for i in range(n):
            Ax = A.dot((X[i, :] - X).T)
            p = Ax * Ax
            p = p.sum(axis=0)
            p = np.exp(-p)
            p[i] = 0
            p = p / p.sum()
            index = labels == labels[i]  # 取相同标签的索引
            all = np.dot(p * (X[i, :] - X).T,X[i, :] - X)#(p * (X[i, :] - X).T).dot((X[i, :] - X))
            p1 = p[index]  # 取相同标签的P
            lall = np.dot(p1 * (X[i, :] - X[index]).T,X[i, :] - X[index]) #(p1 * (X[i, :] - X[index]).T).dot((X[i, :] - X[index]))
            der = A.dot(p1.sum() * all - lall)
            A = A + learning_rate * der
            #psum =p1.sum()
            #print(psum)

    return 0


def Euclidean_distance(inst_a, inst_b):
    return np.linalg.norm(inst_a - inst_b)

def distance(inst_a, inst_b):
    Ax=A.dot((inst_a- inst_b).T)
    dist = (Ax*Ax).sum()
    #dist = np.random.rand()  # 随机度量 这是一个错误示范，请删除这行 _(:3 」∠)_
    return dist


# main program here
if __name__ == '__main__':
    train_file = open('dataset/Letter_Recognition/lr_train_'+'0'+'.pkl', 'rb')
    test_file = open('dataset/Letter_Recognition/lr_test_'+'0'+'.pkl', 'rb')
    LR_train = pkl.load(train_file) # tuple of training data
    LR_test = pkl.load(test_file) # tuple of testing data
    train_file.close()
    test_file.close()
        
    train_X = LR_train[0] # instances of training data
    train_Y = LR_train[1] # labels of training data
    test_X = LR_test[0] # instances of testing data
    test_Y = LR_test[1] # labels of testing data
    #train(LR_train)
    #print(A)
    #print(Euclidean_distance(train_X[0],train_X[48]))
    traindata = LR_train
    max_iter = 100
    learning_rate = 0.005
    X = traindata[0]
    labels = traindata[1]
    n, d = X.shape
    global A
    A =  np.zeros((d, d))
    np.fill_diagonal(A, 1)
    psum =0.0
    for it in range(10):
        for i in range(n):
            Ax = A.dot((X[i, :] - X).T)
            p = Ax * Ax
            p = p.sum(axis=0)
            p = np.exp(-p)
            p[i] = 0
            p = p / p.sum()
            index = labels == labels[i]  # 取相同标签的索引
            all = np.dot(p * (X[i, :] - X).T,X[i, :] - X)#(p * (X[i, :] - X).T).dot((X[i, :] - X))
            p1 = p[index]  # 取相同标签的P
            lall = np.dot(p1 * (X[i, :] - X[index]).T,X[i, :] - X[index]) #(p1 * (X[i, :] - X[index]).T).dot((X[i, :] - X[index]))
            der = A.dot(p1.sum() * all - lall)
            A = A + learning_rate * der
            psum =p1.sum()
            #print(psum)
              #print(psum)