# -*- coding: utf-8 -*-
__author__ = 'Kay'

import matplotlib.pyplot as plt
from stock_feature import get_stock_features
from sklearn import linear_model
from sklearn.svm import SVR
import seaborn as sns
sns.set(style='ticks', palette='Set2')


def to_list(li):
    '''
    X -> GRU的数据格式转一般线性回归的格式
    :param li:
    :return:
    '''
    new_list = []
    for i in li:
        one_X = []
        for j in i:
            for k in j:
                one_X.append(k)
        new_list.append(one_X)

    return new_list


def draw_accuracy(real_y, pre_y):
    y = []
    if len(real_y) != len(pre_y):
        print('ERROR: different length.')
        return 0

    # for i in list(range(len(real_y))):
    #     if


def draw_bingo(high_y, low_y, pre_y):
    y = []
    if len(high_y) != len(pre_y) or len(low_y) != len(pre_y):
        print('ERROR: different length.')
        return 0

    # for i in list(range(len(real_y))):
    #     if


def get_test_score(max_len):
    mod = linear_model.Ridge(alpha = 0.5)
    # mod = SVR(C=1.0, epsilon=0.2)
    # mod = linear_model.LinearRegression()
    X, y, X_test, y_test = get_stock_features('../data/data-with-BS.txt', max_len=max_len)

    X = to_list(X)
    X_test = to_list(X_test)

    mod.fit(X, y)

    # print('参数：')
    # print(mod.coef_)
    # print(mod.intercept_)
    # print('测试结果 =', mod.predict(X_test))
    # print('实际结果 =', y_test)
    score = mod.score(X_test, y_test)

    return score


def get_train_score(max_len):
    mod = linear_model.Ridge(alpha = 0.5)
    # mod = SVR(C=1.0, epsilon=0.2)
    # mod = linear_model.LinearRegression()
    X, y, X_test, y_test = get_stock_features('../data/data-with-BS.txt', max_len=max_len)

    X = to_list(X)
    mod.fit(X, y)
    score = mod.score(X, y)

    return score


def main():

    x = []
    y = []
    for i in list(range(1, 21)):
        s = get_train_score(i)
        x.append(i)
        y.append(s)
    print(x)
    print(y)

    plt.plot(x, y, 'r', linewidth=2)
    plt.xlabel('$Days\ of\ lag$', fontsize=15)
    plt.ylabel('$R^2$', fontsize=15)
    plt.title('$Train\ R^2\ of\ Ridge\ regression$', fontsize=15)
    plt.show()
    plt.close()


    x = []
    y = []
    for i in list(range(1, 21)):
        s = get_test_score(i)
        x.append(i)
        y.append(s)
    print(x)
    print(y)

    plt.plot(x, y, 'b', linewidth=2)
    plt.xlabel('$Days\ of\ lag$', fontsize=15)
    plt.ylabel('$R^2$', fontsize=15)
    plt.title('$Test\ R^2\ of\ Ridge\ regression$', fontsize=15)
    plt.show()
    plt.close()

    # plt.savefig('../figure/R^2-of-Ridge-regression.png')


if __name__ == '__main__':
    main()


