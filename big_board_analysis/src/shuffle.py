# -*- coding: utf-8 -*-
__author__ = 'Kay'

import numpy as np
import random
from scipy.stats.stats import pearsonr
import os


def write_mean_var(input_file):
    '''
    计算均值及方差并写入
    :param input_file:
    :return:
    '''
    print('写入均值与标准差 ...')
    raw_data = np.loadtxt(input_file)
    me = raw_data.mean()
    var = raw_data.std()
    print(me, var)
    output_file = input_file + '_mean_var'
    open(output_file, 'w').write(str(me) + ' ' + str(var) + '\n')
    return me, var


def my_sample(input_file, output_dir, n):
    '''
    写入随机采样结果
    :param input_file:
    :return:
    '''
    print('sampling ...')
    rows = np.loadtxt(input_file).tolist()
    result = random.sample(rows, n)
    new_txt = np.array(result)
    rows, cols = new_txt.shape
    for i in range(5):
        for j in range(5, cols):
            pear, sig = pearsonr(new_txt[:, i], new_txt[:, j])
            emotion = j % 5
            lags = int(j / 5)
            output_file = open(output_dir + '/%s_%s_%s_pearson'
                               % (i, emotion, lags), 'a')
            output_file.write(str(pear) + '\n')


def my_shuffle(input_file, output_file):
    '''
    随机打乱
    :param input_file:
    :param output_file:
    :return:
    '''
    txt = np.loadtxt(input_file)
    rows, cols = txt.shape
    new_txt = list()
    for i in range(cols):
        col = txt[:, i].tolist()
        random.shuffle(col)
        print(col)
        new_txt.append(col)
    new_txt = np.array(new_txt).transpose()
    np.savetxt(output_file, new_txt, fmt='%.8f')


input_file = '../data/proportion/normalization.txt'
# output_dir = '../data/abs/sample'

'''
打乱
'''
output_file = '../data/proportion/shuffle.txt'
my_shuffle(input_file, output_file)


'''
采样
'''
# for i in range(100):
#     my_sample(input_file, output_dir, 150)


'''
计算均值与标准差
'''
# for fi in os.listdir(output_dir):
#     if fi.endswith('pearson'):
#         me, var = write_mean_var(os.path.join(output_dir, fi))




