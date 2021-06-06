# -*- coding: utf-8 -*-
__author__ = 'Kay'

import numpy as np
from sklearn.linear_model import LogisticRegression
# from sklearn import cross_validation
from sklearn import grid_search
from evaluate import get_precision_3, get_precision_2
from sklearn.datasets import load_svmlight_file

def translate_list(li):
    '''
    转换列表中的值，因为precision_3需要1，2，3，而预测结果为-1，0，1
    :param li:
    :return:
    '''
    new_li = []
    for e in li:
        if e == -1: new_li.append(0)
        elif e == 0: new_li.append(1)
        elif e == 1: new_li.append(2)
        else: print('error!'); break
    return new_li


def LRM_para(X, y, c):
    lr = LogisticRegression(penalty='l2', tol=0.01, solver='lbfgs', multi_class='multinomial')

    # 创建参数序列
    parameters = {'C': np.logspace(c, c, num=1, base=2)}

    # 开始调参
    clf = grid_search.GridSearchCV(lr, parameters, cv=5)
    clf.fit(X, y)
    # clf.fit(raw_data[:, 6::5], raw_data[:, i]) # just sadness

    # 显示交叉验证的结果（最好的参数下）
    # print(clf.estimator)
    print('交叉验证准确率:', clf.best_score_, clf.best_params_, '\n')

    # 预测及评估（自己的分类评估函数）
    y_hat = clf.best_estimator_.predict(raw_data[:, 5:])
    # y_hat = translate_list(y_hat); y = translate_list(y)
    # get_precision_3(y, y_hat)
    get_precision_2(y, y_hat)


def LRM(X, y):
    lr = LogisticRegression(penalty='l2', tol=0.01, solver='lbfgs', multi_class='multinomial')

    # 创建参数序列
    parameters = {'C': np.logspace(-16, 16, num=33, base=2)}
    # parameters = {'C': np.logspace(-8, 8, num=17, base=2)}
    # parameters = {'C': np.logspace(-4, 4, num=9, base=2)}

    # 开始调参
    clf = grid_search.GridSearchCV(lr, parameters, cv=5)
    clf.fit(X, y)
    # clf.fit(raw_data[:, 6::5], raw_data[:, i]) # just sadness

    # 显示交叉验证的结果（最好的参数下）
    # print(clf.estimator)
    print('交叉验证准确率:', clf.best_score_, clf.best_params_, '\n')

    # 预测及评估（自己的分类评估函数）
    y_hat = clf.best_estimator_.predict(X)
    # y_hat = translate_list(y_hat); y = translate_list(y)
    get_precision_3(y, y_hat)
    # get_precision_2(y, y_hat)


# 读取数据
# raw_data = np.loadtxt('../data/proportion/classify/eq_3_class.txt')
raw_data = np.loadtxt('../data/proportion/classify/kmeans_3_class.txt')
# raw_data = np.loadtxt('../data/proportion/classify/2_class.txt')
# X = raw_data[:, 5:]
X = raw_data[:, [8, 13, 18, 23, 28]]
print(X)

# 遍历5个目标值，进行调参和交叉验证结果的输出
for i in range(5):
# for i in range(2):
    y = raw_data[:, i]
    LRM(X, y)
    # print('begin ---------------------------------------------------------')
    # for i in range(-8, 8):
    #     LRM_para(X, y, i)


# 建立逻辑斯特回归基本模型
#
# lr.fit(raw_data[:, 5:], raw_data[:, 1])
# print(lr.score(raw_data[:, 5:], raw_data[:, 1]))
# lr = LogisticRegression(penalty='l1', tol=0.01, solver='lbfgs', multi_class='multinomial')
# clf.fit(raw_data[:, 5:], raw_data[:, 0])
# print(raw_data[:, 0])
# print(clf.predict(raw_data[:, 5:]))
# print(clf.score(raw_data[:, 5:], raw_data[:, 0]))
# print(clf.coef_)
# print(scores.mean(), scores.std())






