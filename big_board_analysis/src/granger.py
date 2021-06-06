# -*- coding: utf-8 -*-
__author__ = 'Kay'


import numpy as np
import pandas as pd
import pandas.stats.var as var
from statsmodels.tsa.stattools import grangercausalitytests
import re


def tell_the_cause(input_file):
    '''
    方便Eviews计算结果中提取出有用的信息
    :param input_file:
    :return:
    '''
    print(input_file)
    li = []
    for line in open(input_file):
        line = line.strip()
        words = line.split('\t')
        if len(words) == 4:
            if words[3] not in ['Prob.', ' NA'] and float(words[3]) <= 0.05 and words[0].startswith('X') and words[0][-3] != 'X':
                if float(words[3]) >= 0.01:
                    mark = '*'
                elif float(words[3]) >= 0.001:
                    mark = '**'
                else:
                    mark = '***'
                li.append(line + ' ' + mark)
    li.sort()
    for l in li:
        print(l)


def granger_diff(in_name, out_name):
    '''
    为平稳化做一阶差分
    :param in_name:
    :param out_name:
    :return:
    '''
    txt = np.loadtxt(in_name, dtype=float)
    new_txt = np.diff(txt.T)
    np.savetxt(out_name, new_txt.T, fmt='%.4f', delimiter=',')


def my_granger(in_name):
    data = pd.read_csv(in_name)

    for target in ['close', 'open', 'high', 'low', 'volume']:
        for mood in ['anger', 'disgust', 'joy', 'sadness', 'fear']:
            d = np.asarray(data[[target, mood]])
            print('>>>', target, mood)
            grangercausalitytests(d, maxlag=5)


def analyse_my_granger(in_name):
    lags = None
    for line in open(in_name):
        if line.startswith('>>>'):
            t = line.strip()
        if line.startswith('number of lags (no zero) '):
            lags = int(line.strip()[len('number of lags (no zero) ')])
        if line.startswith('parameter'):
            p_value = float(line.split(',')[1][3:])
            if p_value < 0.05:
                print(t, lags, p_value)


def analyse_r_result(in_name):
    i = 0
    for line in open(in_name):
        if line.startswith('Model 1'):
            i = 1
        else:
            i += 1
        if i == 1:
            # print(line.strip())
            model = line.strip()
        if i == 5:
            try:
                p_value = float(line.strip().split(' ')[-1])
            except:
                p_value = float(line.strip().split(' ')[-2])
                lags = int(model.split(', ')[-1][:-1].split(':')[1])
                if p_value < 0.05 and lags <= 10:
                    # print(model)
                    goal = model.split('Lags(')[1].split(', ')[0]
                    emotion = model.split('Lags(')[2].split(', ')[0]
                    print(goal, '~', emotion, '~', lags, '~', p_value)
                    # print(line.strip())


if __name__ == '__main__':
    # tell_the_cause('../data/granger/ratio/lag4.txt')
    # dir = '/Users/Kay/Project/ML/big_board_analysis/data/granger/'
    # granger_diff(dir + '20141201-20150916_比例_格兰杰_normalization.txt', dir + '20141201-20150916_比例_格兰杰_normalization_diff.txt')
    # my_granger('../data/granger/user_1_regression.txt')
    analyse_my_granger('1.out')

    # analyse_r_result('../data/granger_result.txt')
    # analyse_r_result('../data/granger_result_user_2.txt')
