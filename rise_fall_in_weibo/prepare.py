# -*- coding: utf-8 -*-
__author__ = 'Kay'

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model

sns.set(style="darkgrid")


data = pd.read_csv('data/VIP_day.csv')
keyword_dat = pd.read_csv('data/keyword_results.csv')
# retweet_dat = pd.read_csv('data/sum_retweet_results.csv')
finance_emotion = pd.read_csv('data/finance_emotion.csv')


# 合并关键词后的数据
# data = pd.merge(data, keyword_dat)
# data = pd.merge(data, retweet_dat)
data = pd.merge(data, finance_emotion)
data = pd.merge(data, keyword_dat)


def linear(x, y):
    '''
    线性回归
    :param x:
    :param y:
    :return:
    '''
    y = y.tolist()
    for i in np.arange(len(y)):
        y[i] = math.log(y[i])
    plt.plot([0, 0], [9, 12], 'r-')
    plt.plot(x, y, 'b*')
    plt.xlim(-10, 10)
    plt.xlabel('Rate of return')
    plt.ylabel('Weibo amount')
    # plt.show()
    x = [[i] for i in x.tolist()]

    clf = linear_model.LinearRegression()
    clf.fit(x, y)
    print('参数:', clf.coef_[0], clf.intercept_)
    print('R^2:', clf.score(x, y))



# print('---------- 总量 ----------')
# print(data.close.corr(data.su))
# print(abs(data.close).corr(data.su))
# linear(data.close, data.su)
#
# print('---------- 区分方向 ----------')
# print(data.close[data.close>0].corr(data.su[data.close>0]))
# linear(data.close[data.close>0], data.su[data.close>0])
#
# print(data.close[data.close<0].corr(data.su[data.close<0]))
# linear(data.close[data.close<0], data.su[data.close<0])



# print('---------- abs > 0.1 ----------')
# print(data.close[data.close>0.1].corr(data.su[data.close>0.1]))
# print(data.close[data.close<-0.1].corr(data.su[data.close<-0.1]))
#
# print('---------- abs > 0.2 ----------')
# print(data.close[data.close>0.2].corr(data.su[data.close>0.2]))
# print(data.close[data.close<-0.2].corr(data.su[data.close<-0.2]))
#
# print('---------- abs > 0.4 ----------')
# print(data.close[data.close>0.4].corr(data.su[data.close>0.4]))
# print(data.close[data.close<-0.4].corr(data.su[data.close<-0.4]))
#
# print('---------- abs > 0.6 ----------')
# print(data.close[data.close>0.6].corr(data.su[data.close>0.6]))
# print(data.close[data.close<-0.6].corr(data.su[data.close<-0.6]))
#
# print('---------- 不同情绪 ----------')
# print(data.close.corr(data.anger))
# print(data.close.corr(data.disgust))
# print(data.close.corr(data.joy))
# print(data.close.corr(data.sadness))
# print(data.close.corr(data.fear))
#
# print(data.close[data.close>0].corr(data.anger[data.close>0]))
# print(data.close[data.close>0].corr(data.disgust[data.close>0]))
# print(data.close[data.close>0].corr(data.joy[data.close>0]))
# print(data.close[data.close>0].corr(data.sadness[data.close>0]))
# print(data.close[data.close>0].corr(data.fear[data.close>0]))

# print('---------- 关键词与收益率 ----------')
# print(data.close.corr(data.bull))
# print(data.close.corr(data.bear))
# print(data.close.corr(data.buy))
# print(data.close.corr(data.sell))
# linear(data.close, data.bull)
# linear(data.close, data.bear)
# linear(data.close, data.buy)
# linear(data.close, data.sell)
#
#
# print('---------- 关键词与收益率>0 ----------')
# print(data.close[data.close>0].corr(data.bull[data.close>0]))
# print(data.close[data.close>0].corr(data.bear[data.close>0]))
# print(data.close[data.close>0].corr(data.buy[data.close>0]))
# print(data.close[data.close>0].corr(data.sell[data.close>0]))
# linear(data.close[data.close>0], data.bull[data.close>0])
# linear(data.close[data.close>0], data.bear[data.close>0])
# linear(data.close[data.close>0], data.buy[data.close>0])
# linear(data.close[data.close>0], data.sell[data.close>0])
#
#
# print('---------- 关键词与收益率<0 ----------')
# print(data.close[data.close<0].corr(data.bull[data.close<0]))
# print(data.close[data.close<0].corr(data.bear[data.close<0]))
# print(data.close[data.close<0].corr(data.buy[data.close<0]))
# print(data.close[data.close<0].corr(data.sell[data.close<0]))
# linear(data.close[data.close<0], data.bull[data.close<0])
# linear(data.close[data.close<0], data.bear[data.close<0])
# linear(data.close[data.close<0], data.buy[data.close<0])
# linear(data.close[data.close<0], data.sell[data.close<0])

# print('---------- 转发率与收益率 ----------')
# print(data.close.corr(data.retweet / data.su))
# linear(data.close, data.retweet / data.su)
#
#
# print('---------- 转发率与收益率>0 ----------')
# print(data.close[data.close>0].corr((data.retweet / data.su)[data.close>0]))
# linear(data.close[data.close>0], (data.retweet / data.su)[data.close>0])
#
#
# print('---------- 转发率与收益率<0 ----------')
# print(data.close[data.close<0].corr((data.retweet / data.su)[data.close<0]))
# linear(data.close[data.close<0], (data.retweet / data.su)[data.close<0])


# 金融情绪数据和社交情绪数据的关系

def relation(x, y):
    '''
    线性关系
    '''
    plt.xlabel(x.name)
    plt.ylabel(y.name)
    plt.scatter(x, y, alpha=0.7)
    # plt.show()
    plt.savefig('figure/{}-{}.pdf'.format(y.name, x.name))
    plt.close()

    x_fit = x.values.reshape((-1, 1))
    clf = linear_model.LinearRegression().fit(x_fit, y)

    print('-' * 20 + x.name + ', ' + y.name + '-' * 20)
    # print(clf.coef_)
    # print(clf.intercept_)
    print('参数:', clf.coef_[0], clf.intercept_)
    print('R^2:', clf.score(x_fit, y))
    print('\n')



# targets = [data.DMI, data.AR, data.BR, data.ROC]
data.RSbull = data.bull / data.su
data.RSbull.name = 'proportion of bull'

data.RSbear = data.bull / data.su
data.RSbear.name = 'proportion of bear'

data.RSbuy = data.bull / data.su
data.RSbuy.name = 'proportion of buy'

data.RSsell = data.bull / data.su
data.RSsell.name = 'proportion of sell'

# targets = [data.bull, data.bear, data.buy, data.sell]
targets = [data.RSbull, data.RSbear, data.RSbuy, data.RSsell]



for target in targets:
    # print(data.su.corr(target))
    # relation(data.su, target)

    print(data.anger.corr(target))
    relation(data.anger, target)

    print(data.disgust.corr(target))
    relation(data.disgust, target)

    print(data.joy.corr(target))
    relation(data.joy, target)

    print(data.sadness.corr(target))
    relation(data.sadness, target)

    print(data.fear.corr(target))
    relation(data.fear, target)

    data.RJA = (data.joy / data.anger).apply(math.log)
    data.RJA.name = 'RJA'
    print(data.RJA.corr(target))
    relation(data.RJA, target)

    data.RJD = (data.joy / data.disgust).apply(math.log)
    data.RJD.name = 'RJD'
    print(data.RJD.corr(target))
    relation(data.RJD, target)

    data.RJS = (data.joy / data.sadness).apply(math.log)
    data.RJS.name = 'RJS'
    print(data.RJS.corr(target))
    relation(data.RJS, target)

    data.RJF = (data.joy / data.fear).apply(math.log)
    data.RJF.name = 'RJF'
    print(data.RJF.corr(target))
    relation(data.RJF, target)
