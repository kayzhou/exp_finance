__author__ = 'Kay Zhou'

# coding: utf-8

# In[57]:

import math
import numpy as np
import pandas as pd
from matplotlib import ticker
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns

sns.set(style="darkgrid")
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))


def load_data():
    # # 情绪与股市数据
    # raw_data = pd.read_csv('data/VIP_day.csv')
    # # 关键词数据，涨跌买卖
    # keyword_data = pd.read_csv('data/keyword_results.csv')
    # # 转发与总量
    # sum_retweet_data = pd.read_csv('data/sum_retweet_results.csv')
    # # 合并数据
    # data = pd.merge(pd.merge(raw_data, keyword_data), sum_retweet_data)
    # # print(data)
    # data.to_csv('data/2014-2016_data.csv')

    data = pd.read_csv('data/2014-2016_data.csv')
    return data


# In[58]:

def linear(x, y):
    '''
    线性回归
    :param x:
    :param y:
    :return:
    '''
    print('相关性：', x.corr(y))
    y = y.tolist()
    # for i in np.arange(len(y)):
    #     y[i] = math.log(y[i])
    # plt.plot([0, 0], [9, 12], 'r-')
    # plt.plot(x, y, 'b*')
    # plt.xlim(-10, 10)
    # plt.show()
    x = [[i] for i in x.tolist()]

    clf = LinearRegression()
    clf.fit(x, y)
    print('参数:', clf.coef_, clf.intercept_)
    print('R^2:', clf.score(x, y))
    print()


# In[59]:

def scatter_and_line(x_n, y_n, x, y, x_label, y_label, fig_name):
    '''
    牛熊市下用户行为的差异
    :param x_n: 收益率小于0
    :param y_n:
    :param x: 收益率大于0
    :param y:
    :return:
    '''

    def linear(input_x, input_y):
        '''
        线性回归
        :param x:
        :param y:
        :return: 拟合参数
        '''
        clf = LinearRegression()
        clf.fit([[i] for i in input_x.tolist()], input_y.tolist())
        return clf.coef_[0], clf.intercept_

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.yaxis.set_major_formatter(formatter)

    ax.scatter(x, y, alpha=0.6)
    ax.scatter(x_n, y_n, alpha=0.6)

    # ax.plot([0, 0], [0, 20])

    # 画拟合直线
    a, b = linear(x, y)
    p_x = [min(x), max(x)]
    p_y = [min(x) * a + b, max(x) * a + b]
    ax.plot(p_x, p_y, '-', color=(197 / 255, 58 / 255, 50 / 255), linewidth=2.0)

    a, b = linear(x_n, y_n)
    p_x = [min(x_n), max(x_n)]
    p_y = [min(x_n) * a + b, max(x_n) * a + b]
    ax.plot(p_x, p_y, '-', color=(197 / 255, 58 / 255, 50 / 255), linewidth=2.0)

    plt.xlabel(x_label, fontsize=18)
    plt.ylabel(y_label, fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.set_rasterized(True)
    plt.savefig(fig_name, dpi=300)
    plt.close('all')


def scatter_and_line_volume(x, y, x_label, y_label, fig_name):
    '''
    牛熊市下用户行为的差异
    :param x:
    :param y:
    :return:
    '''

    def linear(input_x, input_y):
        '''
        线性回归
        :param x:
        :param y:
        :return: 拟合参数
        '''
        clf = LinearRegression()
        clf.fit([[i] for i in input_x.tolist()], input_y.tolist())
        return clf.coef_[0], clf.intercept_

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax.yaxis.set_major_formatter(formatter)

    ax.scatter(x, y, alpha=0.5)

    # 画拟合直线
    a, b = linear(x, y)
    p_x = [min(x), max(x)]
    p_y = [min(x) * a + b, max(x) * a + b]
    ax.plot(p_x, p_y, '-', color=(197 / 255, 58 / 255, 50 / 255), linewidth=2.0)

    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    ax.set_rasterized(True)
    plt.savefig(fig_name, dpi=300)
    plt.close('all')


def return_and_weibo(data):
    print('---------- 总量 ----------')
    linear(data.close, data.su)
    linear(abs(data.close), data.su)

    print('---------- 区分方向 ----------')
    linear(data.close[data.close > 0], data.su[data.close > 0])
    linear(data.close[data.close < 0], data.su[data.close < 0])

    # 收益率与总量
    # 散点图及线性拟合
    x_n = data.close[data.close < 0]
    y_n = data.su[data.close < 0]
    x = data.close[data.close > 0]
    y = data.su[data.close > 0]
    scatter_and_line(x_n, y_n, x, y, '$r_{t}$', '$the\ amount\ of\ tweets$', 'figure/r_amount.pdf')

    # 收益率等级不同

    # In[63]:
    print('---------- abs > 0.1 ----------')
    linear(data.close[abs(data.close) > 0.1], data.su[abs(data.close) > 0.1])
    linear(data.close[data.close > 0.1], data.su[data.close > 0.1])
    linear(data.close[data.close < -0.1], data.su[data.close < -0.1])

    x_n = data.close[data.close < -0.1]
    y_n = data.su[data.close < -0.1]
    x = data.close[data.close > 0.1]
    y = data.su[data.close > 0.1]
    scatter_and_line(x_n, y_n, x, y, '$rate\ of\ return$', '$the\ amount\ of\ tweets$', 'figure/amount_0.1.pdf')

    print('---------- abs > 0.5 ----------')
    linear(data.close[abs(data.close) > 0.5], data.su[abs(data.close) > 0.5])
    linear(data.close[data.close > 0.5], data.su[data.close > 0.5])
    linear(data.close[data.close < -0.5], data.su[data.close < -0.5])

    x_n = data.close[data.close < -0.5]
    y_n = data.su[data.close < -0.5]
    x = data.close[data.close > 0.5]
    y = data.su[data.close > 0.5]
    scatter_and_line(x_n, y_n, x, y, '$rate\ of\ return$', '$the\ amount\ of\ tweets$', 'figure/amount_0.5.pdf')

    print('---------- abs > 1.0 ----------')
    linear(data.close[abs(data.close) > 1], data.su[abs(data.close) > 1])
    linear(data.close[data.close > 1], data.su[data.close > 1])
    linear(data.close[data.close < -1], data.su[data.close < -1])

    x_n = data.close[data.close < -1]
    y_n = data.su[data.close < -1]
    x = data.close[data.close > 1]
    y = data.su[data.close > 1]
    scatter_and_line(x_n, y_n, x, y, '$rate\ of\ return$', '$the\ amount\ of\ tweets$', 'figure/amount_1.0.pdf')

    print('---------- abs > 2.0 ----------')
    linear(data.close[abs(data.close) > 2], data.su[abs(data.close) > 2])
    linear(data.close[data.close > 2], data.su[data.close > 2])
    linear(data.close[data.close < -2], data.su[data.close < -2])

    x_n = data.close[data.close < -2]
    y_n = data.su[data.close < -2]
    x = data.close[data.close > 2]
    y = data.su[data.close > 2]
    scatter_and_line(x_n, y_n, x, y, '$rate\ of\ return$', '$the\ amount\ of\ tweets$', 'figure/amount_2.0.pdf')

    # In[64]:

    # 不同情绪相关性
    print('---------- 绝对情绪 ----------')
    linear(data.close, data.anger)
    linear(data.close, data.disgust)
    linear(data.close, data.joy)
    linear(data.close, data.sadness)
    linear(data.close, data.fear)

    print('---------- 收益率>0，绝对情绪 ----------')
    linear(data.close[data.close > 0], data.anger[data.close > 0])
    linear(data.close[data.close > 0], data.disgust[data.close > 0])
    linear(data.close[data.close > 0], data.joy[data.close > 0])
    linear(data.close[data.close > 0], data.sadness[data.close > 0])
    linear(data.close[data.close > 0], data.fear[data.close > 0])

    print('---------- 收益率<0，绝对情绪 ----------')
    linear(data.close[data.close < 0], data.anger[data.close < 0])
    linear(data.close[data.close < 0], data.disgust[data.close < 0])
    linear(data.close[data.close < 0], data.joy[data.close < 0])
    linear(data.close[data.close < 0], data.sadness[data.close < 0])
    linear(data.close[data.close < 0], data.fear[data.close < 0])

    # In[65]:

    # 收益率与绝对情绪
    # 散点图及线性拟合

    print("愤怒")
    x_n = data.close[data.close < 0]
    y_n = data.anger[data.close < 0]
    x = data.close[data.close > 0]
    y = data.anger[data.close > 0]
    scatter_and_line(x_n, y_n, x, y, '$rate\ of\ return$', '$anger$', 'figure/abs_anger.pdf')

    print("厌恶")
    x_n = data.close[data.close < 0]
    y_n = data.disgust[data.close < 0]
    x = data.close[data.close > 0]
    y = data.disgust[data.close > 0]
    scatter_and_line(x_n, y_n, x, y, '$rate\ of\ return$', '$disgust$', 'figure/abs_disgut.pdf')

    print("高兴")
    x_n = data.close[data.close < 0]
    y_n = data.joy[data.close < 0]
    x = data.close[data.close > 0]
    y = data.joy[data.close > 0]
    scatter_and_line(x_n, y_n, x, y, '$rate\ of\ return$', '$joy$', 'figure/abs_joy.pdf')

    print("低落")
    x_n = data.close[data.close < 0]
    y_n = data.sadness[data.close < 0]
    x = data.close[data.close > 0]
    y = data.sadness[data.close > 0]
    scatter_and_line(x_n, y_n, x, y, '$rate\ of\ return$', '$sadness$', 'figure/abs_sadness.pdf')

    print("恐惧")
    x_n = data.close[data.close < 0]
    y_n = data.fear[data.close < 0]
    x = data.close[data.close > 0]
    y = data.fear[data.close > 0]
    scatter_and_line(x_n, y_n, x, y, '$rate\ of\ return$', '$fear$', 'figure/abs_fear.pdf')

    # In[66]:

    # 收益率与相对情绪
    # 相对情绪 log( 1+joy / 1+neg )

    d_close = data.close
    rela_anger = ((1 + data.anger) / (1 + data.joy)).apply(math.log)
    rela_disgust = ((1 + data.disgust) / (1 + data.joy)).apply(math.log)
    rela_sadness = ((1 + data.sadness) / (1 + data.joy)).apply(math.log)
    rela_fear = ((1 + data.fear) / (1 + data.joy)).apply(math.log)

    linear(d_close, rela_anger)
    linear(d_close, rela_disgust)
    linear(d_close, rela_sadness)
    linear(d_close, rela_fear)

    print('---------- 收益率>0，相对情绪 ----------')
    linear(d_close[data.close > 0], rela_anger[data.close > 0])
    linear(d_close[data.close > 0], rela_disgust[data.close > 0])
    linear(d_close[data.close > 0], rela_sadness[data.close > 0])
    linear(d_close[data.close > 0], rela_fear[data.close > 0])

    print('---------- 收益率<0，相对情绪 ----------')
    linear(d_close[data.close < 0], rela_anger[data.close < 0])
    linear(d_close[data.close < 0], rela_disgust[data.close < 0])
    linear(d_close[data.close < 0], rela_sadness[data.close < 0])
    linear(d_close[data.close < 0], rela_fear[data.close < 0])

    # # 收益率分布
    # d_close.hist(bins=30)
    # plt.show()
    #
    # # 相对情绪的分布
    # fig = plt.figure(0)
    # rela_anger.hist(bins=30, alpha=0.8)
    # plt.show()
    # plt.close(0)
    # fig = plt.figure(0)
    # rela_disgust.hist(bins=30, alpha=0.8)
    # plt.show()
    # plt.close(0)
    # fig = plt.figure(0)
    # rela_sadness.hist(bins=30, alpha=0.8)
    # plt.show()
    # plt.close(0)
    # fig = plt.figure(0)
    # rela_fear.hist(bins=30, alpha=0.8)
    # plt.show()
    # plt.close(0)


    print("愤怒")
    x_n = data.close[data.close < 0]
    y_n = rela_anger[data.close < 0]
    x = data.close[data.close > 0]
    y = rela_anger[data.close > 0]
    scatter_and_line(x_n, y_n, x, y, '$rate\ of\ return$', '$anger$', 'figure/rel_anger.pdf')

    print("厌恶")
    x_n = data.close[data.close < 0]
    y_n = rela_disgust[data.close < 0]
    x = data.close[data.close > 0]
    y = rela_disgust[data.close > 0]
    scatter_and_line(x_n, y_n, x, y, '$rate\ of\ return$', '$joy$', 'figure/rel_disgust.pdf')

    print("低落")
    x_n = data.close[data.close < 0]
    y_n = rela_sadness[data.close < 0]
    x = data.close[data.close > 0]
    y = rela_sadness[data.close > 0]
    scatter_and_line(x_n, y_n, x, y, '$rate\ of\ return$', '$sadness$', 'figure/rel_sadness.pdf')

    print("恐惧")
    x_n = data.close[data.close < 0]
    y_n = rela_fear[data.close < 0]
    x = data.close[data.close > 0]
    y = rela_fear[data.close > 0]
    scatter_and_line(x_n, y_n, x, y, '$rate\ of\ return$', '$fear$', 'figure/rel_fear.pdf')

    # In[70]:

    print('---------- 关键词与收益率 ----------')
    linear(data.close, data.bull)
    linear(data.close, data.bear)
    linear(data.close, data.buy)
    linear(data.close, data.sell)

    print('---------- 关键词与收益率>0 ----------')
    linear(data.close[data.close > 0], data.bull[data.close > 0])
    linear(data.close[data.close > 0], data.bear[data.close > 0])
    linear(data.close[data.close > 0], data.buy[data.close > 0])
    linear(data.close[data.close > 0], data.sell[data.close > 0])

    print('---------- 关键词与收益率<0 ----------')
    linear(data.close[data.close < 0], data.bull[data.close < 0])
    linear(data.close[data.close < 0], data.bear[data.close < 0])
    linear(data.close[data.close < 0], data.buy[data.close < 0])
    linear(data.close[data.close < 0], data.sell[data.close < 0])

    # In[71]:

    # 收益率与关键词词频
    # 散点图及线性拟合

    print("涨")
    x_n = data.close[data.close < 0]
    y_n = data.bull[data.close < 0]
    x = data.close[data.close > 0]
    y = data.bull[data.close > 0]
    scatter_and_line(x_n, y_n, x, y, '$rate\ of\ return$', '$rise$', 'figure/word_rise.pdf')

    print("跌")
    x_n = data.close[data.close < 0]
    y_n = data.bear[data.close < 0]
    x = data.close[data.close > 0]
    y = data.bear[data.close > 0]
    scatter_and_line(x_n, y_n, x, y, '$rate\ of\ return$', '$fall$', 'figure/word_fall.pdf')

    print("买")
    x_n = data.close[data.close < 0]
    y_n = data.buy[data.close < 0]
    x = data.close[data.close > 0]
    y = data.buy[data.close > 0]
    scatter_and_line(x_n, y_n, x, y, '$rate\ of\ return$', '$buy$', 'figure/word_buy.pdf')

    print("卖")
    x_n = data.close[data.close < 0]
    y_n = data.sell[data.close < 0]
    x = data.close[data.close > 0]
    y = data.sell[data.close > 0]
    scatter_and_line(x_n, y_n, x, y, '$rate\ of\ return$', '$sell$', 'figure/word_sell.pdf')

    # In[72]:

    print('---------- 转发率与收益率 ----------')
    linear(data.close, data.retweet / data.su_stat)
    print('---------- 转发率与收益率>0 ----------')
    linear(data.close[data.close > 0], (data.retweet / data.su_stat)[data.close > 0])
    print('---------- 转发率与收益率<0 ----------')
    linear(data.close[data.close < 0], (data.retweet / data.su_stat)[data.close < 0])

    print('---------- 转发数与收益率 ----------')
    linear(data.close, data.retweet)
    print('---------- 转发数与收益率>0 ----------')
    linear(data.close[data.close > 0], (data.retweet)[data.close > 0])
    print('---------- 转发数与收益率<0 ----------')
    linear(data.close[data.close < 0], (data.retweet)[data.close < 0])

    print('---------- 原创数与收益率 ----------')
    linear(data.close, data.su - data.retweet)
    print('---------- 原创数与收益率>0 ----------')
    linear(data.close[data.close > 0], (data.su_stat - data.retweet)[data.close > 0])
    print('---------- 原创数与收益率<0 ----------')
    linear(data.close[data.close < 0], (data.su_stat - data.retweet)[data.close < 0])

    # 收益率与转发原创行为
    # 散点图及线性拟合

    print("转发率")
    x_n = data.close[data.close < 0]
    y_n = (data.retweet / data.su_stat)[data.close < 0]
    x = data.close[data.close > 0]
    y = (data.retweet / data.su_stat)[data.close > 0]
    scatter_and_line(x_n, y_n, x, y, '$rate\ of\ return$', '$rate\ of\ retweeting$', 'figure/retweet_rate.pdf')

    print("转发数")
    x_n = data.close[data.close < 0]
    y_n = data.retweet[data.close < 0]
    x = data.close[data.close > 0]
    y = data.retweet[data.close > 0]
    scatter_and_line(x_n, y_n, x, y, '$rate\ of\ return$', '$amount\ of\ retweeted\ tweets$',
                     'figure/retweet_retweeted.pdf')

    print("原创数")
    x_n = data.close[data.close < 0]
    y_n = (data.su_stat - data.retweet)[data.close < 0]
    x = data.close[data.close > 0]
    y = (data.su_stat - data.retweet)[data.close > 0]
    scatter_and_line(x_n, y_n, x, y, '$rate\ of\ return$', '$amount\ of\ original\ tweets$',
                     'figure/retweet_original.pdf')


def volume_and_weibo(data):
    print('\n --------------- 成交量 ---------------')
    # 试试成交量，很神奇，跟量并没有很直接的关系

    # 总量
    print('总量')
    linear(data.volume, data.su)
    scatter_and_line_volume(data.volume, data.su, x_label='$volume$', y_label='$amount\ of\ tweets$',
                            fig_name='figure/volume_amount.pdf')

    # 情绪
    rela_anger = ((1 + data.joy) / (1 + data.anger)).apply(math.log)
    rela_disgust = ((1 + data.joy) / (1 + data.disgust)).apply(math.log)
    rela_sadness = ((1 + data.joy) / (1 + data.sadness)).apply(math.log)
    rela_fear = ((1 + data.joy) / (1 + data.fear)).apply(math.log)

    print('相对情绪')
    linear(data.volume, rela_anger)
    scatter_and_line_volume(data.volume, rela_anger, x_label='$volume$', y_label='$anger$',
                            fig_name='figure/volume_anger.pdf')

    linear(data.volume, rela_disgust)
    scatter_and_line_volume(data.volume, rela_disgust, x_label='$volume$', y_label='$disgust$',
                            fig_name='figure/volume_disgust.pdf')

    linear(data.volume, rela_sadness)
    scatter_and_line_volume(data.volume, rela_sadness, x_label='$volume$', y_label='$sadness$',
                            fig_name='figure/volume_sadness.pdf')

    linear(data.volume, rela_fear)
    scatter_and_line_volume(data.volume, rela_fear, x_label='$volume$', y_label='$fear$',
                            fig_name='figure/volume_fear.pdf')

    # 关键词
    print('关键词')
    linear(data.volume, data.bull)
    scatter_and_line_volume(data.volume, data.bull, x_label='$volume$', y_label='$rise$',
                            fig_name='figure/volume_bull.pdf')
    linear(data.volume, data.bear)
    scatter_and_line_volume(data.volume, data.bear, x_label='$volume$', y_label='$fall$',
                            fig_name='figure/volume_bear.pdf')
    linear(data.volume, data.buy)
    scatter_and_line_volume(data.volume, data.bull, x_label='$volume$', y_label='$buy$',
                            fig_name='figure/volume_buy.pdf')
    linear(data.volume, data.sell)
    scatter_and_line_volume(data.volume, data.sell, x_label='$volume$', y_label='$sell$',
                            fig_name='figure/volume_sell.pdf')

    print("转发率")
    linear(data.volume, data.retweet / data.su_stat)
    scatter_and_line_volume(data.volume, data.retweet / data.su_stat, x_label='$volume$',
                            y_label='$rate\ of\ retweeting$', fig_name='figure/volume_rate_retweet.pdf')

    print("转发数")
    linear(data.volume, data.retweet)
    scatter_and_line_volume(data.volume, data.retweet, x_label='$volume$', y_label='$amount\ of\ retweeted\ tweets$',
                            fig_name='figure/volume_retweet.pdf')

    print("原创数")
    linear(data.volume, data.su_stat - data.retweet)
    scatter_and_line_volume(data.volume, data.su_stat - data.retweet, x_label='$volume$',
                            y_label='$amount\ of\ original\ tweets$', fig_name='figure/volume_original.pdf')


if __name__ == '__main__':
    data = load_data()
    # return_and_weibo(data)
    # volume_and_weibo(data)
