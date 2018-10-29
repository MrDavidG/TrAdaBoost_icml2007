# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: icml07.py
@time: 2018/10/20 7:24 PM

"""

import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import datetime


def trainWeightedClassifier(data_training, labels_training, weights):
    model = svm.LinearSVC(verbose=0, max_iter=5000)
    # model = tree.DecisionTreeClassifier(criterion="gini", max_features="log2", splitter="random")
    model.fit(data_training, labels_training, sample_weight=weights)
    return model

def loadData(path):
    file = open(path)
    labels = list()
    data = list()
    for line in file:
        labels = labels + [int(line.split('\t')[0])]
        data = data + [[int(i) for i in line.split('\t')[1:]]]
    return np.array(data), np.array(labels)


def predict_final(model_list, beta_t_list, data, label, threshold):
    res = 1
    for i in range(len(model_list)):
        h_t = model_list[i].predict([data])[0]
        res = res / beta_t_list[i] ** h_t
    if res >= threshold:
        label_predict = 1
    else:
        label_predict = 0
    if label_predict == label:
        return 1
    else:
        return 0


def error_calculate(model, training_data_target, training_labels_target, weights):
    total = np.sum(weights)
    labels_predict = model.predict(training_data_target)
    error = np.sum(weights / total * np.abs(labels_predict - training_labels_target))
    return error


def TrAdaBoost(N=100):
    # 数据处理
    training_data_source, training_labels_source = loadData('./datasets/mushroom_tapering')
    data_target, labels_target = loadData('./datasets/mushroom_enlarging')

    training_data_target, test_data_target, training_labels_target, test_labels_target = train_test_split(data_target,
                                                                                                          labels_target,
                                                                                                          test_size=0.25)

    # 合成训练数据
    training_data = np.r_[training_data_source, training_data_target]
    training_labels = np.r_[training_labels_source, training_labels_target]

    # 对比试验 baseline方法
    svm_0 = svm.LinearSVC(verbose=0, max_iter=5000)
    svm_0.fit(training_data, training_labels)
    print '——————————————————————————————————————————————'
    print '训练数据用目标域和源域的情况'
    print 'The mean accuracy is ' + str(svm_0.score(test_data_target, test_labels_target))
    print 'The error rate is ' + str(1 - svm_0.score(test_data_target, test_labels_target))

    svm_1 = svm.LinearSVC(verbose=0, max_iter=5000)
    svm_1.fit(training_data_target, training_labels_target)
    print '——————————————————————————————————————————————'
    print '训练数据仅用目标域的情况'
    print 'The mean accuracy is ' + str(svm_1.score(test_data_target, test_labels_target))
    print 'The error rate is ' + str(1 - svm_1.score(test_data_target, test_labels_target))

    svm_2 = svm.LinearSVC(verbose=0, max_iter=5000)
    svm_2.fit(training_data_source, training_labels_source)
    print '——————————————————————————————————————————————'
    print '训练数据仅用源域的情况'
    print 'The mean accuracy is ' + str(svm_2.score(test_data_target, test_labels_target))
    print 'The error rate is ' + str(1 - svm_2.score(test_data_target, test_labels_target))
    print '——————————————————————————————————————————————'

    # 训练主循环
    n_source = len(training_data_source)
    m_target = len(training_data_target)
    # 初始化权重
    weights = np.concatenate((np.ones(n_source)/n_source, np.ones(m_target)/m_target))
    beta_t_list = list()
    model_list = list()
    beta = 1.0 / (1.0 + np.sqrt(2 * np.log(n_source) / N))
    for t in range(N):
        p_t = weights / sum(weights)
        model = trainWeightedClassifier(training_data, training_labels, p_t)

        # 加权的错误率
        error_self = error_calculate(model, training_data_target, training_labels_target, weights[-m_target:])

        # 计算参数
        if error_self > 0.5:
            error_self = 0.5
        elif error_self == 0:
            t = N
            break

        beta_t = error_self / (1 - error_self)
        print datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ': ' + '第' + str(t) + '轮的加权错误率为: ' + str(
            error_self)

        # 源域
        for i in range(n_source):
            if model.predict([training_data_source[i]])[0] != training_labels_source[i]:
                weights[i] = weights[i] * beta

        # 目标域
        for i in range(m_target):
            if model.predict([training_data_target[i]])[0] != training_labels_target[i]:
                weights[i + n_source] = weights[i + n_source] / beta_t

        # 记录当前的参数
        beta_t_list += [beta_t]
        model_list += [model]

    # 测试最后输出的模型
    count_accu = 0
    index_half = np.ceil(N / 2)
    threshold = 1
    for beta_t in beta_t_list[index_half:]:
        threshold = threshold / np.sqrt(beta_t)

    for i in range(len(test_data_target)):
        count_accu += predict_final(model_list[index_half:], beta_t_list[index_half:], test_data_target[i],
                                    test_labels_target[i], threshold)
    error_final = 1.0 - count_accu / float(len(test_data_target))
    print datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ': ' + '模型最后的准确率为: ' + str(1 - error_final)
    print datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ': ' + '模型最后的错误率为: ' + str(error_final)


if __name__ == '__main__':
    TrAdaBoost()
