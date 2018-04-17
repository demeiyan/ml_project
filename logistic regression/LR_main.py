# -*- coding: utf-8 -*-
"""
Created on 2018/4/14 20:39

@author: dmyan
使用二分类来实现多分类问题,OvR(一对其余)
"""
import numpy as np


def train_valid_split(feature, label, valid_size=0.1):
    train_indices = []
    valid_indices = []
    for i in range(5):
        index = np.where(label == (i+1))[0]
        np.random.shuffle(index)
        rand = np.random.choice(index.shape[0], int(index.shape[0]*valid_size), replace=False)
        valid_indices.append(index[rand].tolist())
        index = np.delete(index, rand)
        train_indices.append(index.tolist())
    train_indices = [item for sublist in train_indices for item in sublist]
    valid_indices = [item for sublist in valid_indices for item in sublist]
    return feature[train_indices], label[train_indices], feature[valid_indices], label[valid_indices]


def smote(x, y, cls, k):
    label_counts = []
    for i in range(5):
        label_counts.append((y == (i+1)).sum())
    all_samples = x.shape[0]
    label_sample = np.array([int(i == cls) for i in y])
    n_samples = label_sample.sum()
    remian = all_samples - n_samples
    # print(n_samples, remian, sum(label_counts))
    positive_sample = x[np.where(label_sample == 1)[0]]
    # print(positive_sample.shape)

    if n_samples > remian:
        sampling_rate = int(n_samples/remian) + 1
        negative_indices = np.where(label_sample == 0)[0]
        np.random.shuffle(negative_indices)
        # print(negative_indices.shape[0])
        distance_vector = np.zeros(negative_indices.shape[0])
        negative_sample = np.zeros((negative_indices.shape[0]*sampling_rate, 11))
        negative_count = 0
        for i in range(negative_indices.shape[0]):
            for j in range(negative_indices.shape[0]):
                distance_vector[j] = np.linalg.norm(x[negative_indices[i]]-x[negative_indices[j]])
            sampling_knn = distance_vector.argsort()[0:k]
            sampling_nn = np.random.choice(sampling_knn, sampling_rate, replace=False)
            for j in range(sampling_nn.shape[0]):
                for k in range(x.shape[1]):
                    negative_sample[negative_count][k] = x[i][k] + np.random.rand()*(x[negative_indices[j]][k] - x[i][k])
                negative_count += 1
        # print(negative_sample[1000:3000])
        synthetic_feature = np.concatenate((positive_sample, negative_sample), axis=0)
        synthetic_label = np.concatenate((label_sample, np.zeros(negative_sample.shape[0])), axis=0)
        return synthetic_feature, synthetic_label
    else:
        negative_sample = np.zeros(0)
        label_sample = np.ones(positive_sample.shape[0])
        for i in range(5):
            rand = np.random.choice(np.where(y == (i+1))[0], int(label_counts[i]*n_samples/all_samples), replace=False)
            if i == 0:
                negative_sample = x[rand]
            else:
                negative_sample = np.append(negative_sample, x[rand], axis=0)
        negative_sample = np.array(negative_sample)
        synthetic_feature = np.concatenate((positive_sample, negative_sample), axis=0)
        synthetic_label = np.concatenate((label_sample, np.zeros(negative_sample.shape[0])), axis=0)
        # print(synthetic_feature, synthetic_label)
        return synthetic_feature, synthetic_label
    # return synthetic_feature, synthetic_label


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


def binary_classification(x, y, one, belta, learn_rate):
    y = np.array([int(i == one) for i in y])
    der = sigmoid(belta, x.T) - y # sigmoid(belta, x.T) - y
    der = np.reshape(der, (y.shape[0], 1))
    der = np.sum(x*der, axis=0)
    belta = belta - learn_rate*der
    return belta


if __name__ == '__main__':
    min_batch = 64
    learn_rate = 0.01
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
    # split data to train_data and valid_data
    x_train, label_train, x_valid, label_valid = train_valid_split(x_train, label_train, 0.1)
    datasize = label_train.shape[0]
    belta = np.zeros((5, 11))

    #belta[0] = binary_classification(x_train[0 * min_batch:(0 + 1) * min_batch], label_train[0 * min_batch:(0 + 1) * min_batch], 1, belta[0], learn_rate)
    # for epoch in range(100):
    #     print(belta[0])
    #     for i in range(int(datasize/min_batch)):
    #         belta[0] = binaryclassification(x_train[i*min_batch:(i+1)*min_batch], label_train[i*min_batch:(i+1)*min_batch], 5, belta[0], learn_rate)

    for i in range(5):
        # x_train, label_train =
        # x, y = smote(x_train, label_train, i+1, 15)
        # print(x,y)
        for epoch in range(1000):
            for j in range(int(datasize/min_batch)):
                belta[i] = binary_classification(x_train[j*min_batch:(j+1)*min_batch], label_train[j*min_batch:(j+1)*min_batch], i+1, belta[i], learn_rate)

        result = np.where(sigmoid(belta[i], x_train.T) >= 0.5)[0]
        label = np.where(label_train == (i+1))[0]
        count = 0
        for index in result:
            if index in label:
                count += 1
        loss = label.shape[0] - count
        print(str(i)+' loss :'+str(loss/label.shape[0]))
        # loss = (result != label_train).sum()/label_train.shape[0]*100

    # 统计各类样本数目
    # label_counts = []
    # for i in range(5):
    #     label_counts.append((label_train == (i+1)).sum())
    #
    # print(sum(label_counts), label_counts)
    accuracy = sigmoid(belta, x_valid.T)
    # # 阈值飘移,再缩放(rescaling)
    # for i in range(accuracy.shape[0]):
    #     accuracy[i] = accuracy[i] * ((sum(label_counts) - label_counts[i])/label_counts[i])
    accuracy = np.argmax(accuracy, axis=0) + 1
    print('precision : %.2f%%' % ((accuracy == label_valid).sum()/label_test.shape[0]*100))
    # test
    label_predict = np.argmax(sigmoid(belta, x_test.T), axis=0) + 1
    precision = []
    recall = []
    print('\t\tpredict\ttest\tcorrect')
    for i in range(5):
        predict = label_predict == (i+1)
        test = label_test == (i+1)
        predict_indices = [j for j, x in enumerate(label_predict) if x == (i+1)]
        test_indices = [j for j, x in enumerate(label_test) if x == (i+1)]
        correct = 0
        for index in predict_indices:
            if index in test_indices:
                correct += 1
        print(str(i+1)+' sample ' + str(predict.sum())+'\t'+str(test.sum())+'\t'+str(correct))
        if predict.sum() != 0:
            precision.append(correct/predict.sum()*100)
        else:
            precision.append(0.0)
        recall.append(correct/test.sum()*100)
    accuracy = (label_predict == label_test).sum()/label_test.shape[0]*100
    print('precision\t: %.2f%% %.2f%% %.2f%% %.2f%% %.2f%%' % (precision[0], precision[1], precision[2], precision[3], precision[4]))
    print('recall\t: %.2f%% %.2f%% %.2f%% %.2f%% %.2f%%' % (recall[0], recall[1], recall[2], recall[3], recall[4]))
    print('test result : %.2f%%' % accuracy)
