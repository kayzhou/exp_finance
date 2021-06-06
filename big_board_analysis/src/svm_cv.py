# -*- coding: utf-8 -*-
__author__ = 'Kay'

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from sklearn.externals import joblib


def my_svm(X, y, col):
    clf = svm.SVC()
    # 创建参数序列
    # parameters = {'C': np.logspace(-16, 16, num=33, base=2), 'gamma': np.logspace(-16, 16, num=33, base=2)}
    # parameters = {'C': np.logspace(-8, 8, num=17, base=2), 'gamma': np.logspace(-8, 8, num=17, base=2)}
    # parameters = {'C': np.logspace(-4, 4, num=9, base=2), 'gamma': np.logspace(-4, 4, num=9, base=2)}
    parameters = {'C': np.logspace(-5, 5, num=11, base=2), 'gamma': np.logspace(-5, 5, num=11, base=2)}
    # print(parameters)
    # 开始调参
    clf = GridSearchCV(clf, parameters, cv=5)
    clf.fit(X, y)
    # clf.fit(raw_data[:, 6::5], raw_data[:, i]) # just sadness

    # 显示交叉验证的结果（最好的参数下）
    # print(clf.best_estimator)
    print('交叉验证准确率:', clf.best_score_, clf.best_params_)

    # 预测及评估（自己的分类评估函数）
    y_hat = clf.best_estimator_.predict(X)
    # print(y)
    # print(y_hat)
    print('accuracy:', accuracy_score(y, y_hat))
    print('f1-score:', f1_score(y, y_hat, average='macro'))

    # joblib.dump(clf.best_estimator_, '../model/20170410/五分类_{}.mod'.format(col))


if __name__ == '__main__':

    # 读取数据
    raw_data = np.loadtxt('../data/proportion/训练数据/user_2_true.csv')
    # raw_data = np.loadtxt('../data/proportion/20141201-20150916_比例_三分类_true.txt')
    # raw_data = np.loadtxt('../data/proportion/classify/kmeans_3_class.txt')

    X = raw_data[:, 5:]
    # 遍历5个目标值，进行调参和交叉验证结果的输出
    for i in range(5):
        print(i)
        y = raw_data[:, i]
        my_svm(X, y, str(i))
