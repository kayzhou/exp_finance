# -*- coding: utf-8 -*-
__author__ = 'Kay'

'''
SVM 回归
'''

from sklearn.svm import SVR
from sklearn.svm import NuSVR
import numpy as np
from sklearn import grid_search

'''
Additionally, negative values of R2 may occur when fitting non-linear functions to data.
In cases where negative values arise,
the mean of the data provides a better fit to the outcomes than do the fitted function values,
according to this particular criterion.
'''


def svr_cv(raw_data):
    '''
    svr做交叉验证及网格调参
    :param raw_data: 原始数据
    :return: 打印最好的预测器及拟合优度
    '''
    X = raw_data[:, 5:30]
    for i in range(5):
        y = raw_data[:, i]
        svr = SVR()
        # clf = NuSVR()
        parameters = {'C': np.arange(0.5, 10, 0.5), 'epsilon': np.arange(0.1, 1, 0.1),
                      'gamma': np.logspace(-6, -1, 10)}
        clf = grid_search.GridSearchCV(svr, param_grid=parameters, cv=5)
        clf.fit(X, y)
        print('预测模型:', clf.best_estimator_)
        print('R-squared:', clf.best_score_)

        # y_hat = clf.predict(X)
        # print(y_hat)


def svr(raw_data, i):
    '''
    在自身数据上做训练及测试，网格调参
    :param raw_data:
    :return:
    '''
    X = raw_data[:, 5:30]
    y = raw_data[:, i]
    # clf = SVR()
    # clf = SVR(C=3.0, epsilon=0.1, gamma=0.002) # 收盘
    # clf = SVR(C=0.5, epsilon=0.5, gamma=10) # 开盘
    # clf = SVR(C=1.5, epsilon=0.4, gamma=0.1) # 最高
    # clf = SVR(C=7.0, epsilon=0.1, gamma=0.002) # 最低
    clf = SVR(C=6.5, epsilon=0.2, gamma=0.02) # 成交量
    # clf = NuSVR()
    clf.fit(X, y)
    print('预测模型:', clf)
    print('R-squared:', clf.score(X, y))

    # y_hat = clf.predict(X)
    # print(y_hat)


if __name__ == '__main__':
    raw_data = np.loadtxt('/Users/Kay/Project/ML/big_board_analysis/data/proportion/回归.csv', delimiter=",")
    # raw_data = np.loadtxt('/Users/Kay/Project/ML/big_board_analysis/data/proportion/normalization.txt')
    svr_cv(raw_data)
    # svr(raw_data, 4)
