# -*- coding: utf-8 -*-
__author__ = 'Kay'

import numpy as np
from sklearn.externals import joblib
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score, f1_score

if __name__ == '__main__':

    raw_data = np.loadtxt('../data/proportion/测试数据/user_0_true.csv')
    X = raw_data[:, 5:]
    for i in range(5):
        print('-' * 20, i, '-' * 20)
        y = raw_data[:, i]
        clf = joblib.load('../model/20170410/user_0_{}.mod'.format(i))
        print('预测结果 =', clf.predict(X))
        print('实际结果 =', y)
        print('score =', clf.score(X, y))
        # print('f1 score =', f1_score(y, clf.predict(X), average='macro'))
        print()
