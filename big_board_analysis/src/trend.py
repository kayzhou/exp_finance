# -*- coding: utf-8 -*-
__author__ = 'Kay'


import numpy as np


def get_trend_rise_fall(li):
    '''
    :param li:
    :return: 获取趋势
    '''
    trend = list()
    for i in range(len(li)):
        trend.append( 1 if float(li[i]) >= 0 else 0 )
    return trend

def get_trend(li):
    '''
    :param li:
    :return: 获取趋势
    '''
    trend = list()
    for i in range(1, len(li)):
        trend.append( 1 if float(li[i]) >= float(li[i - 1]) else 0 )
    return trend

def get_precision(li1, li2, lags=1):
    '''
    :param li1: 真实趋势
    :param li2: 预测趋势
    :return:
    '''
    bingo = 0.0
    for i in range(len(li1) - 1 - lags):
        if li1[i + 1 + lags] == li2[i]:
            bingo += 1
    return bingo / (len(li1) - 1 - lags)

def get_precision_dont_care(li1, li2, lags=1):
    '''
    :param li1: 真实趋势
    :param li2: 预测趋势
    :return:
    '''
    bingo = 0.0
    for i in range(len(li1) - lags):
        if li1[i + lags] == li2[i]:
            bingo += 1
    return bingo / (len(li1) - lags)

def rise_or_fall(li, threshold, bigger=True):
    result = list()
    for l in li:
        if bigger:
            result.append(1 if float(l) >= threshold else 0)
        else:
            result.append(0 if float(l) >= threshold else 1)

    return result

def dont_care():
    input_file = '/Users/Kay/Project/ML/correlation/data/20141201-20150916_绝对数据_格兰杰.txt'
    txt = np.loadtxt(input_file)
    rows, cols = txt.shape
    new_txt = list()
    real = []
    emotion = []

    # 控制提前天数，需要预测天数向右移动
    for i in range(0, 5):
        col = txt[:, i]
        trend = get_trend(col)
        print(trend)
        real.append(trend)

    for i in range(5, cols):
        col = txt[:, i]
        trend = get_trend(col)
        print(trend)
        emotion.append(trend)

    for i in range(len(real)):
        for j in range(len(emotion)):
            for lags in range(1, 6):
                print(i, j, lags, get_precision(real[i], emotion[j], lags))


def care_different_trend():
    '''
    因为大盘等已经是涨跌幅，符号就代表着趋势，计算趋势的时需要不同对待。
    :return:
    '''
    input_file = '/Users/Kay/Project/ML/correlation/data/20141201-20150916_绝对数据_格兰杰.txt'
    txt = np.loadtxt(input_file)
    rows, cols = txt.shape
    new_txt = list()
    real = []
    emotion = []

    # 控制提前天数，需要预测天数向右移动
    for i in range(0, 4):
        col = txt[:, i]
        trend = get_trend_rise_fall(col)
        print(trend)
        real.append(trend)

    amount_trend = get_trend(txt[:, 4])

    for i in range(5, cols):
        col = txt[:, i]
        trend = get_trend(col)
        print(trend)
        emotion.append(trend)

    for i in range(len(real)):
        for j in range(len(emotion)):
            for lags in range(1, 6):
                print(i, j, lags, get_precision(real[i], emotion[j], lags))

    # 交易量与涨跌幅不同，需要特殊处理
    for j in range(len(emotion)):
        for lags in range(1, 6):
            print(4, j, lags, get_precision_dont_care(amount_trend, emotion[j], lags))


if __name__ == '__main__':
    care_different_trend()
    # dont_care()
