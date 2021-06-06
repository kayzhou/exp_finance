# -*- coding: utf-8 -*-
__author__ = 'Kay'

from sklearn import datasets
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import r2_score

# boston = datasets.load_boston()
# lr = LinearRegression()
# lr.fit(boston.data, boston.target)
# print(boston.data, boston.target)
# predictions = lr.predict(boston.data)
#
# print(lr.coef_)
# print(predictions)
# print(type(lr.coef_))

def linear_regression_cv(X, y):
    lr = LinearRegression()
    score = cross_val_score(lr, X, y, cv=5)
    print('CV R2 =', score)
    print('The mean of CV R2 =', score.mean())


def linear_regression(X, y):
    lr = LinearRegression()
    lr.fit(X, y)
    y_hat = lr.predict(X)
    r2 = r2_score(y, y_hat)
    print('线性回归模型：', lr)
    print('r2 =', r2)


def select_column(raw_data, indexes):
    '''
    按indexes选择列数据
    :param raw_data:
    :param indexes:
    :return:
    '''
    new_data = np.array(raw_data[:, indexes[0]])
    for i, index in enumerate(indexes):
        if i == 0: continue
        # print(new_data, raw_data[:, index])
        new_data = np.hstack((new_data, raw_data[:, index]))
    return new_data.reshape(len(raw_data), len(indexes))


if __name__ == '__main__':
    print('test')
    exit(-1)
    raw_data = np.loadtxt('/Users/Kay/Project/ML/big_board_analysis/data/proportion/回归.csv', delimiter=",")
    # X = raw_data[:, 5:30]; y = raw_data[:, 4]
    X = select_column(raw_data, [13, 14, 28, 10, 18, 22]); y = raw_data[:, 4]
    linear_regression_cv(X, y)
    # print(X.shape)
    # print(X)
    # print(y.shape)
    # linear_regression(X, y)