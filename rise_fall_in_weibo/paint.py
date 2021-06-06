__author__ = 'Kay Zhou'

import math
import numpy as np
import pandas as pd
from matplotlib import ticker
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
import warnings

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
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


def paint_r_emotion(fig_name):
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

    def scatter_and_line(ax, x, y, x_n, y_n, x_label, y_label):
        ax.scatter(x_n, y_n, alpha=0.5)
        ax.scatter(x, y, alpha=0.5)

        # 画拟合直线
        a, b = linear(x, y)
        p_x = [min(x), max(x)]
        p_y = [min(x) * a + b, max(x) * a + b]
        ax.plot(p_x, p_y, '-', color=(197 / 255, 58 / 255, 50 / 255), linewidth=2.0)

        a, b = linear(x_n, y_n)
        p_x = [min(x_n), max(x_n)]
        p_y = [min(x_n) * a + b, max(x_n) * a + b]
        ax.plot(p_x, p_y, '-', color=(197 / 255, 58 / 255, 50 / 255), linewidth=2.0)

        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

    data = load_data()
    rela_anger = ((1 + data.anger) / (1 + data.joy)).apply(math.log)
    rela_disgust = ((1 + data.disgust) / (1 + data.joy)).apply(math.log)
    rela_sadness = ((1 + data.sadness) / (1 + data.joy)).apply(math.log)
    rela_fear = ((1 + data.fear) / (1 + data.joy)).apply(math.log)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(221)
    ax.yaxis.set_major_formatter(formatter)
    print("愤怒")
    x_n = data.close[data.close < 0]
    y_n = rela_anger[data.close < 0]
    x = data.close[data.close > 0]
    y = rela_anger[data.close > 0]
    scatter_and_line(ax, x_n, y_n, x, y, '$r_{t}$', '$DAJ$')

    ax = fig.add_subplot(222)
    ax.yaxis.set_major_formatter(formatter)
    print("厌恶")
    x_n = data.close[data.close < 0]
    y_n = rela_disgust[data.close < 0]
    x = data.close[data.close > 0]
    y = rela_disgust[data.close > 0]
    scatter_and_line(ax, x_n, y_n, x, y, '$r_{t}$', '$DDJ$')

    ax = fig.add_subplot(223)
    ax.yaxis.set_major_formatter(formatter)
    print("低落")
    x_n = data.close[data.close < 0]
    y_n = rela_sadness[data.close < 0]
    x = data.close[data.close > 0]
    y = rela_sadness[data.close > 0]
    scatter_and_line(ax, x_n, y_n, x, y, '$r_{t}$', '$DSJ$')

    ax = fig.add_subplot(224)
    ax.yaxis.set_major_formatter(formatter)
    print("恐惧")
    x_n = data.close[data.close < 0]
    y_n = rela_fear[data.close < 0]
    x = data.close[data.close > 0]
    y = rela_fear[data.close > 0]
    scatter_and_line(ax, x_n, y_n, x, y, '$r_{t}$', '$DFJ$')

    plt.tight_layout()
    plt.savefig(fig_name, dpi=300)


def paint_v_emotion(fig_name):
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

    def scatter_and_line(ax, x, y, x_label, y_label):
        ax.scatter(x, y, alpha=0.4)

        # 画拟合直线
        a, b = linear(x, y)
        p_x = [min(x), max(x)]
        p_y = [min(x) * a + b, max(x) * a + b]
        ax.plot(p_x, p_y, '-', color=(197 / 255, 58 / 255, 50 / 255), linewidth=2.0)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

    data = load_data()
    rela_anger = ((1 + data.anger) / (1 + data.joy)).apply(math.log)
    rela_disgust = ((1 + data.disgust) / (1 + data.joy)).apply(math.log)
    rela_sadness = ((1 + data.sadness) / (1 + data.joy)).apply(math.log)
    rela_fear = ((1 + data.fear) / (1 + data.joy)).apply(math.log)

    x = data.volume
    fig = plt.figure(figsize=(10, 6))

    ax = fig.add_subplot(221)
    ax.yaxis.set_major_formatter(formatter)
    print("愤怒")
    scatter_and_line(ax, x, rela_anger, '$v_{t}$', '$DAJ$')

    ax = fig.add_subplot(222)
    ax.yaxis.set_major_formatter(formatter)
    print("厌恶")
    scatter_and_line(ax, x, rela_disgust, '$v_{t}$', '$DDJ$')

    ax = fig.add_subplot(223)
    ax.yaxis.set_major_formatter(formatter)
    print("低落")
    scatter_and_line(ax, x, rela_sadness, '$v_{t}$', '$DSJ$')

    ax = fig.add_subplot(224)
    ax.yaxis.set_major_formatter(formatter)
    print("恐惧")
    scatter_and_line(ax, x, rela_fear, '$v_{t}$', '$DFJ$')

    plt.tight_layout()
    plt.savefig(fig_name, dpi=300)


def paint_r_word(fig_name):
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

    def scatter_and_line(ax, x, y, x_n, y_n, x_label, y_label):
        ax.scatter(x_n, y_n, alpha=0.5)
        ax.scatter(x, y, alpha=0.5)

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

        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

    data = load_data()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(221)
    ax.yaxis.set_major_formatter(formatter)
    x_n = data.close[data.close < 0]
    y_n = data.bull[data.close < 0]
    x = data.close[data.close > 0]
    y = data.bull[data.close > 0]
    scatter_and_line(ax, x_n, y_n, x, y, '$r_{t}$', '$bullish$')

    ax = fig.add_subplot(222)
    ax.yaxis.set_major_formatter(formatter)
    x_n = data.close[data.close < 0]
    y_n = data.bear[data.close < 0]
    x = data.close[data.close > 0]
    y = data.bear[data.close > 0]
    scatter_and_line(ax, x_n, y_n, x, y, '$r_{t}$', '$bearish$')

    ax = fig.add_subplot(223)
    ax.yaxis.set_major_formatter(formatter)
    x_n = data.close[data.close < 0]
    y_n = data.buy[data.close < 0]
    x = data.close[data.close > 0]
    y = data.buy[data.close > 0]
    scatter_and_line(ax, x_n, y_n, x, y, '$r_{t}$', '$buy$')

    ax = fig.add_subplot(224)
    ax.yaxis.set_major_formatter(formatter)
    x_n = data.close[data.close < 0]
    y_n = data.sell[data.close < 0]
    x = data.close[data.close > 0]
    y = data.sell[data.close > 0]
    scatter_and_line(ax, x_n, y_n, x, y, '$r_{t}$', '$sell$')

    plt.tight_layout()
    plt.savefig(fig_name, dpi=300)


def paint_v_word(fig_name):
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

    def scatter_and_line(ax, x, y, x_label, y_label):
        ax.scatter(x, y, alpha=0.4)

        # 画拟合直线
        a, b = linear(x, y)
        p_x = [min(x), max(x)]
        p_y = [min(x) * a + b, max(x) * a + b]
        ax.plot(p_x, p_y, '-', color=(197 / 255, 58 / 255, 50 / 255), linewidth=2.0)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

    data = load_data()

    fig = plt.figure(figsize=(10, 6))
    x = data.volume

    ax = fig.add_subplot(221)
    ax.yaxis.set_major_formatter(formatter)
    scatter_and_line(ax, x, data.bull, '$v_{t}$', '$bullish$')

    ax = fig.add_subplot(222)
    ax.yaxis.set_major_formatter(formatter)
    scatter_and_line(ax, x, data.bear, '$v_{t}$', '$bearish$')

    ax = fig.add_subplot(223)
    ax.yaxis.set_major_formatter(formatter)
    scatter_and_line(ax, x, data.buy, '$v_{t}$', '$buy$')

    ax = fig.add_subplot(224)
    ax.yaxis.set_major_formatter(formatter)
    scatter_and_line(ax, x, data.sell, '$v_{t}$', '$sell$')

    plt.tight_layout()
    plt.savefig(fig_name, dpi=300)


def paint_r_retweet(fig_name):
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

    def scatter_and_line(ax, x, y, x_n, y_n, x_label, y_label):
        ax.scatter(x_n, y_n, alpha=0.5)
        ax.scatter(x, y, alpha=0.5)

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

        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

    data = load_data()
    fig = plt.figure(figsize=(14, 4))
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))

    ax = fig.add_subplot(131)
    print("转发数")
    x_n = data.close[data.close < 0]
    y_n = data.retweet[data.close < 0]
    x = data.close[data.close > 0]
    y = data.retweet[data.close > 0]
    ax.yaxis.set_major_formatter(formatter)
    scatter_and_line(ax, x_n, y_n, x, y, '$r_{t}$', '$amount\ of\ RT$')

    ax = fig.add_subplot(132)
    print("原创数")
    x_n = data.close[data.close < 0]
    y_n = (data.su_stat - data.retweet)[data.close < 0]
    x = data.close[data.close > 0]
    y = (data.su_stat - data.retweet)[data.close > 0]
    ax.yaxis.set_major_formatter(formatter)
    scatter_and_line(ax, x_n, y_n, x, y, '$r_{t}$', '$amount\ of\ creating$'),

    ax = fig.add_subplot(133)
    print("转发率")
    x_n = data.close[data.close < 0]
    y_n = (data.retweet / data.su_stat)[data.close < 0]
    x = data.close[data.close > 0]
    y = (data.retweet / data.su_stat)[data.close > 0]
    ax.yaxis.set_major_formatter(formatter)
    scatter_and_line(ax, x_n, y_n, x, y, '$r_{t}$', '$rate\ of\ RT$')
    plt.tight_layout()
    plt.savefig(fig_name, dpi=300)


def paint_v_retweet(fig_name):
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

    def scatter_and_line(ax, x, y, x_label, y_label):
        ax.scatter(x, y, alpha=0.4)

        # 画拟合直线
        a, b = linear(x, y)
        p_x = [min(x), max(x)]
        p_y = [min(x) * a + b, max(x) * a + b]
        ax.plot(p_x, p_y, '-', color=(197 / 255, 58 / 255, 50 / 255), linewidth=2.0)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

    data = load_data()

    x = data.volume
    fig = plt.figure(figsize=(6, 12))

    ax = fig.add_subplot(311)
    ax.yaxis.set_major_formatter(formatter)
    scatter_and_line(ax, x, data.retweet, '$v_{t}$', '$amount\ of\ RT$')

    ax = fig.add_subplot(312)
    ax.yaxis.set_major_formatter(formatter)
    scatter_and_line(ax, x, data.su_stat - data.retweet, '$v_{t}$', '$amount\ of\ creating$')

    ax = fig.add_subplot(313)
    rate = data.retweet / data.su_stat
    ax.yaxis.set_major_formatter(formatter)
    scatter_and_line(ax, x, rate, '$v_{t}$', '$rate\ of\ RT$')

    plt.tight_layout()
    plt.savefig(fig_name, dpi=300)


def paint_v(fig_name):
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

    def scatter_and_line(ax, x, y, x_label, y_label):
        ax.scatter(x, y, alpha=0.4)

        # 画拟合直线
        a, b = linear(x, y)
        p_x = [min(x), max(x)]
        p_y = [min(x) * a + b, max(x) * a + b]
        ax.plot(p_x, p_y, '-', color=(197 / 255, 58 / 255, 50 / 255), linewidth=2.0)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

    data = load_data()

    x = data.volume
    fig = plt.figure(figsize=(8, 8))

    rela_anger = ((1 + data.anger) / (1 + data.joy)).apply(math.log)
    rela_sadness = ((1 + data.sadness) / (1 + data.joy)).apply(math.log)

    ax = fig.add_subplot(321)
    ax.yaxis.set_major_formatter(formatter)
    scatter_and_line(ax, x, data.buy, '$v_{t}$', '$buy$')

    ax = fig.add_subplot(322)
    ax.yaxis.set_major_formatter(formatter)
    scatter_and_line(ax, x, data.sell, '$v_{t}$', '$sell$')

    ax = fig.add_subplot(323)
    ax.yaxis.set_major_formatter(formatter)
    print("愤怒")
    scatter_and_line(ax, x, rela_anger, '$v_{t}$', '$DAJ$')

    ax = fig.add_subplot(324)
    ax.yaxis.set_major_formatter(formatter)
    print("低落")
    scatter_and_line(ax, x, rela_sadness, '$v_{t}$', '$DSJ$')

    ax = fig.add_subplot(325)
    ax.yaxis.set_major_formatter(formatter)
    scatter_and_line(ax, x, data.retweet, '$v_{t}$', '$amount\ of\ RT$')

    ax = fig.add_subplot(326)
    rate = data.retweet / data.su_stat
    ax.yaxis.set_major_formatter(formatter)
    scatter_and_line(ax, x, rate, '$v_{t}$', '$rate\ of\ RT$')

    plt.tight_layout()
    plt.savefig(fig_name, dpi=300)


if __name__ == '__main__':
    # paint_r_word('figure/r_word.pdf')
    # paint_r_emotion('figure/r_rela_emotion.pdf')
    # paint_r_retweet('figure/r_retweet.pdf')

    # paint_v_word('figure/v_word.pdf')
    # paint_v_emotion('figure/v_rela_emotion.pdf')
    # paint_v_retweet('figure/v_retweet.pdf')

    paint_v('v_all.pdf')
