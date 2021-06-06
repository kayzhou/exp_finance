# coding: utf-8
__author__ = 'Kay'

import numpy as np
import sys

'''
评估模型性能
'''


def get_class(li):
    re = list()
    for l in li:
        re.append(1 if l >= 0 else 0)
    return re


def effective(t_close, high, low):
    bingo = 0.0
    for i in range(len(t_close)):
        if low[i] <= t_close[i] <= high[i]:
            # print(low[i], t_close[i], high[i]
            bingo += 1
        # elif t_close[i] < low[i]:
        #     print('预测偏小。', low[i], t_close[i], high[i]
        # elif t_close[i] > high[i]:
        #     print('预测偏大。', low[i], t_close[i], high[i]
    print('有效率：', bingo / len(t_close))


def coincide(high, low, t_high, t_low):

    if len(high) != len(t_high):
        print('出错：向量长度不同，无法比较。')
        return -1

    coin_40 = 0.0
    coin_50 = 0.0
    coin_60 = 0.0
    coin_70 = 0.0
    coin_80 = 0.0

    sum_coin = 0.0

    for i in range(len(high)):
        if t_high[i] <= t_low[i] or high[i] <= t_low[i] or low[i] >= t_high[i]:
            coin = 0
            continue
        elif t_high[i] >= high[i] and t_low[i] >= low[i]:
            intersection = high[i] - t_low[i]
            union = t_high[i] - low[i]
            coin = intersection / union
        elif t_high[i] <= high[i] and t_low[i] <= low[i]:
            intersection = t_high[i] - low[i]
            union = high[i] - t_low[i]
            coin = intersection / union
        elif t_high[i] >= high[i] and t_low[i] <= low[i]:
            intersection = high[i] - low[i]
            union = t_high[i] - t_low[i]
            coin = intersection / union
        elif t_high[i] <= high[i] and t_low[i] >= low[i]:
            intersection = t_high[i] - t_low[i]
            union = high[i] - low[i]
            coin = intersection / union
        else:
            print('异常：此组合没有考虑。')

        # print('index:', i, 'coin:', coin
        sum_coin += coin

        if coin >= 0.4:
            coin_40 += 1
        if coin >= 0.5:
            coin_50 += 1
        if coin >= 0.6:
            coin_60 += 1
        if coin >= 0.7:
            coin_70 += 1
        if coin >= 0.8:
            coin_80 += 1

    ave_coin = sum_coin / len(high)

    print('不同重合度的命中率：', coin_40 / len(high), coin_50 / len(high), \
          coin_60 / len(high), coin_70 / len(high), coin_80 / len(high))
    print('平均重合度：', ave_coin)


def get_precision_2(li1, li2):
    '''
    评估二分类结果
    :param li1: 真实数据
    :param li2: 预测数据
    :return:
    '''
    print( '二分类模型评估中 ... ...')
    bingo = 0.0
    TP = FP = FN = TN = 0.0
    if len(li1) != len(li2):
        print( '出错：列表大小不同')
        return False

    for i in range(len(li1)):
        if li1[i] == li2[i]:
            bingo += 1
            if li1[i] > 0: TP += 1
            else: TN += 1
        else:
            if li1[i] > 0: FN += 1
            else: FP += 1

    print( '准确率:', bingo / len(li1), ' 命中数:', bingo, ' 总数:', len(li1))

    # 二分类问题更复杂的解释
    try:
        print( 'TP（真阳性）:', TP / (TP + FN), 'TN（真阴性）:', TN / (FP + TN))
    except:
        print( '全部预测错误！')
    try:
        print( 'FP（伪阳性）:', FP / (FP + TN), 'FN（伪阴性）:', FN / (TP + FN))
    except:
        print( '全部预测正确！')

    print( '正样本:', TP + FN, '负样本:', FP + TN)
    print( '预测为正:', TP + FP, '预测为负:', TN + FN, '\n')

    try:
        real_poti = TP + FN; real_nege = FP + TN
        pred_poti = TP + FP; pred_nege = FN + TN
        precision_poti = TP / real_poti; recall_poti = TP / pred_poti
        precision_nege = TN / real_nege; recall_nege = TN / pred_nege
        F1 = (( 2 * precision_poti * recall_poti / (precision_poti + recall_poti) )
            + ( 2 * precision_nege * recall_nege / (precision_nege + recall_nege) )) / 2
        print( 'F1 =', F1)
    except ZeroDivisionError:
        print( 'pity!')
        F1 = 0
    print('-------------------------------')
    return F1
    # return precision_poti, recall_poti, precision_nege, recall_nege


def get_precision_3(li1, li2):
    '''
    评估三分类结果
    :param li1: 真实数据
    :param li2: 预测数据
    :return:
    '''
    # li1 = translate_list(li1); li2 = translate_list(li2) # 要看文件的分类标签来决定, 很重要
    print( '三分类模型评估中 ... ...')


    bingo = 0.0
    T0 = F0 = T1 = F1 = T2 = F2 = P0 = P1 = P2 = 0.0
    if len(li1) != len(li2):
        print( '出错：列表大小不同')
        print(len(li1), len(li2))
        return False

    for i in range(len(li1)):
        if li1[i] == li2[i]:
            bingo += 1
            if li1[i] == 0: T0 += 1
            elif li1[i] == 1: T1 += 1
            elif li1[i] == 2: T2 += 1
        else:
            if li1[i] == 0: F0 += 1
            elif li1[i] == 1: F1 += 1
            elif li1[i] == 2: F2 += 1

        if li2[i] == 0: P0 += 1
        elif li2[i] == 1: P1 += 1
        elif li2[i] == 2: P2 += 1

    print( '准确率:', bingo / len(li1), ' 命中数:', bingo, ' 总数:', len(li1))


    # 三分类问题更复杂的解释
    try:
        print( '-1_准确率:', T0 / (T0 + F0), '; -1_召回率:', T0 / P0)
    except:
        if T0 + F0 == 0: print( '实际结果中没有-1！')
        else: print( '预测结果中没有-1！')
    try:
        print( '0_准确率:', T1 / (T1 + F1), '; 0_召回率:', T1 / P1)
    except:
        if T1 + F1 == 0: print( '实际结果中没有0！')
        else: print( '预测结果中没有0！')
    try:
        print( '1_准确率:', T2 / (T2 + F2), '; 1_召回率:', T2 / P2)
    except:
        if T2 + F2 == 0: print( '实际结果中没有1！')
        else: print( '预测结果中没有1！')

    print( '-1_样本:', T0 + F0, '; 0_样本:', T1 + F1, '; 1_样本:', T2 + F2)
    print( '-1_预测:', P0, '; 0_预测:', P1, '; 1_预测', P2, '\n')
    try:
        F1 = (( 2 * T0 / (T0 + F0) * T0 / P0 / ( T0 / (T0 + F0) + T0 / P0) )
                    +  ( 2 * T1 / (T1 + F1) * T1 / P1 / ( T1 / (T1 + F1) + T1 / P1) )
                    +  ( 2 * T2 / (T2 + F2) * T2 / P2 / ( T2 / (T2 + F2) + T2 / P2) )) / 3
        print( 'F1 =', F1)
    except ZeroDivisionError:
        print(('pity!'))
        F1 = 0

    print('-------------------------------')
    return F1
    # return T0 / (T0 + F0), T0 / P0, T1 / (T1 + F1), T1 / P1, T2 / (T2 + F2), T2 / P2


def translate_list(li):
    '''
    转换列表中的值，因为precision_3需要1，2，3，而预测结果为-1，0，1
    :param li:
    :return:
    '''
    new_li = []
    for e in li:
        if e == -1: new_li.append(0)
        elif e == 0: new_li.append(1)
        elif e == 1: new_li.append(2)
        else: print('error!'); break
    return new_li


if __name__ == '__main__':

    text = np.loadtxt('data/0109-0803_summary.txt', float)

    close = text[:, 1]
    high = text[:, 2]
    low = text[:, 3]

    t_close = text[:, 4]
    t_high = text[:, 5]
    t_low = text[:, 6]

    # get_precision(get_class(high), get_class(t_high))
    # get_precision(get_class(low), get_class(t_low))
    #
    # effective(t_close, high, low)
    # coincide(high, low, t_high, t_low)


