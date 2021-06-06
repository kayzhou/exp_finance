# -*- coding: utf-8 -*-
__author__ = 'Kay Zhou'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arrow
import datetime
import os
from scipy.stats import pearsonr


def volatility(data, col_name):
    # print(len(data))
    # 取log
    d = np.log(data[col_name])
    # d = data[col_name]
    # 取一阶差分
    df = d.diff()[1:]
    # 取标准差 n-1
    std = df.std()
    # 乘以交易日个数开方，年化波动率
    # rst = std * np.sqrt(len(data))
    # 也可以不乘，论文中的式子
    rst = std
    return rst


def volatility_windows(data, col_name, windows=5):
    # print(len(data))
    # 取log
    rst = []
    d = np.log(data[col_name])
    # d = data[col_name]
    # 取一阶差分
    df = d.diff()[1:]
    if windows == 0:
        return df
    # 取标准差
    for i in range(windows, len(df)):
        # print(i)
        # print(d[i - windows: i].std() * math.sqrt(windows))
        # vol = d[i - windows: i].std() * math.sqrt(windows)
        vol = d[i - windows: i].std()
        rst.append("%.4f" % vol)
        # rst.append(vol)
    return rst


def vol_file(in_file):
    print(in_file)
    data = pd.read_csv(in_file, index_col=None)
    # 选择时间段，论文中使用至2015-12-07
    data = data[data.date <= '2015-12-07']
    # data = data[data.date>='2015-08-01']
    # print(data)
    # print(volatility(data, 'anger'))
    # print(volatility(data, 'disgust'))

    # 总体波动
    print('%.4f' % volatility(data, 'joy'))
    print('%.4f' % volatility(data, 'fear'))
    print('%.4f' % volatility(data, 'RJF'))

    # 移动平均情绪波动
    joy_windows = volatility_windows(data, 'joy', windows=20)
    print(joy_windows)

    fear_windows = volatility_windows(data, 'fear', windows=20)
    print(fear_windows)

    RJF_w = volatility_windows(data, 'fear', windows=20)
    print(RJF_w)

    x = data[data.date <= '2015-12-08'].date[21:]
    x = [datetime.datetime.strptime(i, '%Y-%m-%d') for i in x]
    # print(x)
    plt.cla()
    plt.plot(x, joy_windows, '-', label='joy')
    plt.plot(x, fear_windows, '--', label='fear')
    plt.legend(fontsize='large')
    plt.grid()
    plt.ylabel('MAV', fontsize='large')
    plt.xlabel('Date', fontsize='large')
    plt.yticks(fontsize='large')
    plt.xticks(fontsize='large')
    plt.ylim(0, 0.45)
    # plt.figure()
    out_name = os.path.join('figure', os.path.basename(in_file) + '.eps')
    plt.savefig(out_name, dpi=300)
    print('-' * 20)


def vol_RJF_file(in_file):
    print(in_file)
    data = pd.read_csv(in_file, index_col=None)
    # 选择时间段，论文中使用至2015-12-07
    data = data[data.date <= '2015-12-07']
    # data = data[data.date>='2015-08-01']
    # print(data)
    # print(volatility(data, 'anger'))
    # print(volatility(data, 'disgust'))

    # 总体波动
    print('%.4f' % volatility(data, 'RJF'))

    # 移动平均情绪波动
    RJF_w = volatility_windows(data, 'fear', windows=20)
    print(RJF_w)

    x = data[data.date <= '2015-12-07'].date[21:]
    x = [datetime.datetime.strptime(i, '%Y-%m-%d') for i in x]
    # print(x)
    plt.cla()
    plt.plot(x, RJF_w, '-', label='joy')
    plt.legend(fontsize='large')
    plt.grid()
    plt.ylabel('MAV', fontsize='large')
    plt.xlabel('Date', fontsize='large')
    plt.yticks(fontsize='large')
    plt.xticks(fontsize='large')
    plt.ylim(0, 0.45)
    # plt.figure()
    out_name = os.path.join('figure', os.path.basename(in_file) + '.eps')
    plt.savefig(out_name, dpi=300)
    print('-' * 20)


def vol_sh_file(in_file='data/sh.csv'):
    print(in_file)
    data = pd.read_csv(in_file, index_col=None)
    # 选择时间段，论文中使用至2015-12-07
    data = data[data.date <= '2015-12-07']
    # data = data[data.date>='2015-08-01']
    # print(data)
    # print(volatility(data, 'anger'))
    # print(volatility(data, 'disgust'))

    # 总体波动
    print('%.4f' % volatility(data, 'close'))
    # print(volatility(data, 'sadness'))

    # 移动平均情绪波动
    close_windows = volatility_windows(data, 'close', windows=20)
    print(close_windows)

    x = data[data.date <= '2015-12-07'].date[21:]
    x = [datetime.datetime.strptime(i, '%Y-%m-%d') for i in x]
    # print(x)
    plt.cla()
    plt.plot(x, close_windows, '.-', label='Close', linewidth=1.5)
    # plt.legend(fontsize='large')
    plt.grid()
    plt.ylabel('MAV', fontsize='x-large')
    plt.xlabel('Date', fontsize='x-large')
    plt.yticks(fontsize='x-large')
    plt.xticks(fontsize='x-large')
    # plt.ylim(0, 0.45)
    # plt.figure()
    out_name = os.path.join('figure', os.path.basename(in_file) + '.eps')
    plt.savefig(out_name, dpi=300)
    print('-' * 20)


def corr_sh_and_emotion(in_file):
    print(in_file)
    data = pd.read_csv('data/sh.csv', index_col=None)
    # 选择时间段，论文中使用至2015-12-07
    data = data[data.date <= '2015-12-07']

    # 移动平均情绪波动
    close_w = volatility_windows(data, 'close', windows=20)

    data = pd.read_csv(in_file, index_col=None)
    # 选择时间段，论文中使用至2015-12-07
    data = data[data.date <= '2015-12-07']

    # 移动平均情绪波动
    joy_w = volatility_windows(data, 'joy', windows=20)
    fear_w = volatility_windows(data, 'fear', windows=20)

    print(len(close_w), len(joy_w))
    print(pearsonr(close_w, joy_w))
    print(pearsonr(close_w, fear_w))


if __name__ == '__main__':
    # vol_file('data/emotion_pro.csv')
    #
    # vol_file('data/emotion_foll_level_0_pro.csv')
    # vol_file('data/emotion_foll_level_1_pro.csv')
    # vol_file('data/emotion_foll_level_2_pro.csv')
    #
    # vol_file('data/emotion_verify_0_pro.csv')
    # vol_file('data/emotion_verify_1_pro.csv')
    #
    # vol_file('data/emotion_gender_f_pro.csv')
    # vol_file('data/emotion_gender_m_pro.csv')

    # 计算波动率及移动平均波动率，并且画图
    vol_sh_file()

    # 股市波动率和情绪波动率的相关性
    # corr_sh_and_emotion('data/emotion_pro.csv')
    # corr_sh_and_emotion('data/emotion_foll_level_0_pro.csv')
    # corr_sh_and_emotion('data/emotion_foll_level_1_pro.csv')
    # corr_sh_and_emotion('data/emotion_foll_level_2_pro.csv')
    # corr_sh_and_emotion('data/emotion_gender_f_pro.csv')
    # corr_sh_and_emotion('data/emotion_gender_m_pro.csv')
