# -*- coding: utf-8 -*-
__author__ = 'Kay'

import numpy as np
from sklearn.covariance import EllipticEnvelope
import os

'''
数据预处理，归一化，等频率分类等
'''

def regress_to_classify(in_name, out_name):
    '''
    回归问题转分类问题
    :param in_name:
    :param out_name:
    :param count_of_class:
    :return:
    '''
    txt = np.loadtxt(in_name, 'float')
    new_txt = list()

    # 文本前五列，三分类
    for i in range(5):
        col = txt[:, i]
        order = np.sort(col)
        boundary_1 = order[int(len(order) / 3)]
        boundary_2 = order[int(len(order) / 3) * 2]
        print('boundary 1 =', boundary_1)
        print('boundary 2 =', boundary_2)

        new_col = []
        for j in range(len(col)):
            if col[j] <= boundary_1:
                new_col.append(-1)
            elif col[j] > boundary_1 and col[j] <= boundary_2:
                new_col.append(0)
            else:
                new_col.append(1)
        new_txt.append(new_col)
        c_n1 = 0; c_0 = 0; c_1 = 1

        for j in range(len(new_col)):
            if new_col[j] == -1:
                c_n1 += 1
            if new_col[j] == 0:
                c_0 += 1
            if new_col[j] == 1:
                c_1 += 1
        print(c_n1, c_0, c_1)

    for i in range(5, 30):
        new_col = txt[:, i]
        new_txt.append(new_col)

    new_txt = np.array(new_txt).transpose()
    np.savetxt(out_name, new_txt, fmt='%.4f')


def true_classify(in_name, out_name):
    '''
    真正的分类标签诞生了！
    :param in_name:
    :param out_name:
    :return:
    '''
    out_fi = open(out_name, 'w')

    for line in open(in_name):
        line = line.strip()
        splits = line.split(' ')
        for i in range(5):
            # print(splits[i])
            splits[i] = str(int(float(splits[i])))
        out_fi.write(' '.join(splits) + '\n')


def make_dataset(in_name):
    '''
    创造libsvm的格式
    :param in_name:
    :return:
    '''

    # 五个目标
    for goal_col in range(5):
        out_file = open(os.path.splitext(in_name)[0] + '_GOAL_%s.txt' % goal_col, 'w')

        for line in open(in_name):
            lines = line.strip().split(' ')

            GOAL = str(int(lines[goal_col]))  # -1 可能存在问题
            del lines[0: 5]
            for i in range(0, len(lines)):
                lines[i] = str(i) + ':' + lines[i]

            out_file.write(GOAL + ' ' + ' '.join(lines) + '\n')


def make_libsvm_dataset_class3(in_name, out_name, goal_col):
    '''
    创造libsvm的格式
    :param in_name:
    :param out_name:
    :param goal_col:
    :return:
    '''
    out_fi = open(out_name, 'w')

    for line in open(in_name):
        splits = line.strip().split(' ')

        GOAL = str(int(splits[goal_col]) - 1) # -1 可能存在问题
        del splits[0: 5]
        for i in range(0, len(splits)):
            splits[i] = str(i) + ':' + splits[i]

        out_fi.write(GOAL + ' ' + ' '.join(splits) + '\n')


def make_libsvm_dataset_class2(in_name, out_name, goal_col):
    '''
    创造libsvm的格式
    :param in_name:
    :param out_name:
    :param goal_col:
    :return:
    '''
    out_fi = open(out_name, 'w')

    for line in open(in_name):
        line = line.strip()
        splits = line.split(' ')

        GOAL = str(int(splits[goal_col])) # +1 可能存在问题
        del splits[0: 5]
        for i in range(0, len(splits)):
            splits[i] = str(i) + ':' + splits[i]

        out_fi.write(GOAL + ' ' + ' '.join(splits) + '\n')


def max_min_normalization(input_file, output_file):
    '''
    线性函数转换，最大最小归一化
    :param input_file:
    :param output_file:
    :return:
    '''
    txt = np.loadtxt(input_file, 'float')
    rows, cols = txt.shape
    new_txt = list()

    for i in range(cols):

        # 分类问题不需要对收盘和开盘价格等进行归一化
        if i in [0, 1, 2, 3, 4]:
            print(txt[:, i])
            new_txt.append(txt[:, i])
            continue

        col = txt[:, i]
        max_of_col = max(col); min_of_col = min(col)

        if max_of_col == min_of_col:
            new_col = col
        else:
            new_col = (col - min_of_col) / (max_of_col - min_of_col)

        print(max_of_col, min_of_col)
        new_txt.append(new_col)

    new_txt = np.array(new_txt).transpose()
    np.savetxt('../tmp/out.txt', new_txt, fmt='%.4f')

    true_classify('../tmp/out.txt', output_file)


def remove_surprise(input_file, output_file):
    '''
    删除异常点
    :param input_file:
    :param output_file:
    :return:
    '''
    txt = np.loadtxt(input_file, 'float')
    rows, cols = txt.shape
    new_txt = list()

    for i in range(rows):
        row = txt[i]
        if abs(row[0]) < 3:
            new_txt.append(row)
    new_txt = np.array(new_txt)
    np.savetxt(output_file, new_txt, fmt='%.8f')


def read_max_min(in_name):
    '''
    读入最大最小值
    :param in_name:
    :return:
    '''
    return np.loadtxt(in_name)


def max_min_test(in_name, out_name, max_min_name='../data/max_min.txt'):
    '''
    对测试数据进行归一化，利用训练数据的归一化方法
    :param in_name:
    :param out_name:
    :return:
    '''

    def max_min_normalise(X, max_min):
        '''
        最大最小归一化，利用原有的数据, 以便 max_min_X() 进行调用
        :param X:
        :param in_name:
        :return:
        '''
        new_X = []
        for i, x in enumerate(X):
            if i in list(range(5)):
                new_X.append(x)
            else:
                new_X.append((x - max_min[i - 5][1]) / (max_min[i - 5][0] - max_min[i - 5][1]))
        return new_X

    # 读入训练数据保存的最大最小值
    max_min = read_max_min(max_min_name)

    X = np.loadtxt(in_name)
    new_txt = []
    for line_x in X:
        # print(line_x)
        new_txt.append(max_min_normalise(line_x, max_min))
    np.savetxt('../tmp/out.txt', new_txt, fmt='%.4f')

    true_classify('../tmp/out.txt', out_name)


def max_min_X_2_classification(in_name, out_name):
    '''
    对测试数据进行归一化，利用训练数据的归一化方法, 二分类
    :param in_name:
    :param out_name:
    :return:
    '''

    def max_min_normalise(X, in_name='../data/max_min.txt'):
        '''
        最大最小归一化，利用原有的数据, 以便 max_min_X() 进行调用
        :param X:
        :param in_name:
        :return:
        '''
        max_min = read_max_min(in_name)
        new_X = []
        for i, x in enumerate(X):
            if i in [0, 1, 2, 3, 4]:
                new_X.append(x)
            else:
                new_X.append((x - max_min[i - 5][1]) / (max_min[i - 5][0] - max_min[i - 5][1]))
        return new_X

    X = np.loadtxt(in_name)
    new_txt = []
    for line_x in X:
        print(line_x)
        line_x[0]=1 if line_x[0] >= 0.0 else 0
        line_x[1]=1 if line_x[1] >= 0.0 else 0
        new_txt.append(max_min_normalise(line_x))
    np.savetxt(out_name, new_txt, fmt='%.8f')


if __name__ == '__main__':

    # 训练数据的预处理
    # max_min_normalization('../data/proportion/20141201-20150916_比例_三分类.txt',
    #                       '../data/proportion/20141201-20150916_比例_三分类_归一化.txt')
    # max_min_normalization('../data/proportion/20141201-20150916_比例_二分类.txt',
    #                       '../data/proportion/20141201-20150916_比例_二分类_归一化.txt')


    # make_libsvm_dataset('../data/proportion/20141201-20150916_比例_三分类_归一化_true.txt',
    #                     '../data/proportion/classify/class3_GOAL4.txt', 4)

    # max_min_normalization('../data/proportion/训练数据/user_2.csv',
    #                       '../data/proportion/训练数据/user_2_归一化.csv')


    # 测试数据的预处理
    max_min_test('../data/proportion/测试数据/20150917_20151207_比例_二分类.txt',
                 '../data/proportion/测试数据/20150917_20151207_比例_二分类_true.txt',
                 '../data/max_min.txt')
    # true_classify('../data/proportion/测试数据/20150917-20151207_比例_三分类_归一化.txt',
    #               '../data/proportion/测试数据/20150917-20151207_比例_三分类_true.txt')

    # max_min_test('../data/proportion/测试数据/user_2.csv',
    #              '../data/proportion/测试数据/user_2_归一化.csv',
    #              '../data/max_min_user_2.txt')
    # true_classify('../data/proportion/测试数据/user_2_归一化.csv',
    #               '../data/proportion/测试数据/user_2_true.csv')


    # 做标准数据
    # make_dataset('../data/proportion/测试数据/20150917-20151207_比例_五分类_true.txt')
    # make_dataset('../data/proportion/测试数据/20150917-20151207_比例_七分类_true.txt')
    # make_dataset('../data/proportion/20141201-20150916_比例_七分类_true.txt')
    # max_min_X('../data/proportion/测试数据/20150917-20151207_比例_三分类.txt',
    #           '../data/proportion/测试数据/20150917-20151207_比例_三分类_normal.txt')
    # max_min_X_and_classify('../data/proportion/测试数据/20150917_20151207.txt',
    #                        '../data/proportion/测试数据/20150917-20151207_比例_二分类_normal.txt')


