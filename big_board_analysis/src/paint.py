# -*- coding: utf-8 -*-
__author__ = 'Kay'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def number2emotion(num):
    if num == '0':
        return 'anger'
    elif num == '1':
        return 'disgust'
    elif num == '2':
        return 'joy'
    elif num == '3':
        return 'sadness'
    elif num == '4':
        return 'fear'


def number2goal(num):
    if num == '0':
        return 'close'
    elif num == '1':
        return 'open'
    elif num == '2':
        return 'high'
    elif num == '3':
        return 'low'
    elif num == '4':
        return 'amount'


def draw_raw_data(X1, X2):
    '''
    时间序列展示，支持两组
    :param X1:
    :param X2:
    :return:
    '''
    X = np.arange(len(X1))
    print(X, X1, X2)
    plt.plot(X, X1, color='blue', alpha=0.6, linewidth=1.5)
    plt.plot(X, X2, color='red', alpha=0.6, linewidth=1.5)
    plt.xlim(-10, 200)
    plt.xlabel('time')
    plt.ylabel('value')
    plt.show()


def draw_scatter_plot(X1, X2):
    '''
    画两个向量组成的散点图
    :param X1:
    :param X2:
    :return:
    '''
    plt.scatter(X1, X2, alpha='0.5')
    plt.show()


def draw_one_other_scatter_plot(X, Y):
    '''
    画N向量组成的散点图，情绪和一种预测目标
    :param X1:
    :param X2:
    :return:
    '''
    row_cnt, col_cnt = Y.shape
    k = 1
    for i in range(col_cnt):
        axis = plt.subplot(5, 5, k); k += 1
        axis.set_xticks([]); axis.set_yticks([])
        axis.scatter(Y[:, k-2], X, color='blue', alpha='0.4')

        lr = LinearRegression()
        X_ = [[x] for x in Y[:, k-2]]
        # print(X_)
        Y_ = X
        lr.fit(X_, Y_)
        # x_ = np.arange(0, 1, 0.01)
        # y_ = x_ * lr.coef_[0] + lr.intercepXt_ # 这是常数项
        print(lr.coef_)
        axis.plot(X_, lr.predict(X_), color='r', lw=2)

    plt.show()


def draw_all_scatter_plot(raw_data):
    '''
    画N向量组成的散点图
    :param X1:
    :param X2:
    :return:
    '''
    row_cnt, col_cnt = raw_data.shape
    k = 1
    for i in range(col_cnt):
        for j in range(col_cnt):
            print(i+1, j+1)
            axes = plt.subplot(col_cnt, col_cnt, k); k += 1
            axes.set_xticks([])
            axes.set_yticks([])
            plt.scatter(raw_data[:, i], raw_data[:, j], color='blue', alpha='0.5')
            lr = LinearRegression()
            X = [[x] for x in raw_data[:, i]]
            y = raw_data[:, j]
            lr.fit(X, y)
            print(lr.coef_)
    plt.show()


def draw_curve(input_dir, output_dir, goal):
    raw_data = np.loadtxt('%s/%s.txt' % (input_dir, goal), dtype='float')
    raw_data = raw_data.transpose()
    # raw_data = raw_data.__abs__()

    X = np.arange(1, 6)
    for i in range(len(raw_data)):
        Y = raw_data[i]
        print(X, Y)
        plt.plot(X, raw_data[i], label=number2emotion(str(i)))

    plt.legend(loc='upper right')
    plt.title(number2goal(goal))
    plt.ylabel('correlation')
    plt.xlabel('days')
    plt.ylim(-1, 1)
    plt.savefig('%s/ylim_1_%s.png' % (output_dir, goal)) # 注意更改目标文件
    plt.show()

def get_mean_var(mean_var_dir, goal, emotion, lags):
    '''
    取得均值与标准差
    :param goal:
    :param emotion:
    :param lags:
    :return:
    '''
    fi = open(mean_var_dir + '/%s_%s_%s_pearson_mean_var' % (goal, emotion, lags))
    mean, var = fi.read().strip().split(' ')
    # print(mean, var)
    return float(mean), float(var)


def write_mean_var(mean_var_dir, out_name):
    '''
    写入均值与标准差
    :param goal:
    :param emotion:
    :param lags:
    :return:
    '''
    out_file = open(out_name, 'w')
    out_file.write(' '.join(['goal', 'emotion', 'lags', 'mean', 'max', 'min']) + '\n')
    for goal in range(5):
        for emotion in range(5):
            for lags in range(1, 6):
                print(goal, emotion, lags)
                fi = open(mean_var_dir + '/%d_%d_%d_pearson_mean_var' % (goal, emotion, lags))
                mean, var = fi.read().strip().split(' ')
                out_file.write(' '.join([str(goal), number2emotion(str(emotion)), str(lags), str(mean),
                                         str(float(mean) + float(var)), str(float(mean) - float(var))]) + '\n')
    out_file.close()



def draw_error_bar(mean_var_dir, output_dir, goal):
    for emotion in ['0', '1', '2', '3', '4']:
        X = np.arange(1, 6)
        yerr_up = list()
        Y = list()

        for lags in range(1, 6):
            mean, var = get_mean_var(mean_var_dir, goal, emotion, str(lags))
            print(mean, var)
            Y.append(mean)
            yerr_up.append(var)
            print(mean, mean + var, mean - var)

        plt.errorbar(X, Y, yerr=yerr_up, fmt='o-', label=number2emotion(emotion))

    plt.legend(loc='upper right')
    plt.title(number2goal(goal))
    plt.ylim(-1, 1)
    plt.xlim(0, 6)
    plt.savefig(output_dir + '/ylim_1_error_bar_%s.png' % goal)
    plt.show()


def main():
    draw_curve('/Users/Kay/Project/ML/correlation/data/remove_surprise/pearson/4.txt')
    # draw_curve('../data/low_trend_correlation.txt')


if __name__ == '__main__':
    # for i in range(5):
    #     draw_error_bar('../data/ratio/sample', '../data/ratio/paint', str(i))
    # for i in range(5):
    #     draw_curve('../data/ratio/pearson', '../data/ratio/paint', str(i))
    # draw_raw_data('/Users/Kay/Project/ML/correlation/data/ratio/normalization.txt')

    write_mean_var('/Users/Kay/Project/ML/big_board_analysis/data/proportion/sample', 'mean_max_min.txt')
    # input_file = '../data/proportion/normalization.txt'
    # input_file = '../data/abs/normalization.txt'
    # raw_data = np.loadtxt(input_file, 'float')

    # draw_raw_data(raw_data[:, 4], raw_data[:, 8])

    # draw_scatter_plot(raw_data[:, 4], raw_data[:, 5])
    # draw_scatter_plot(raw_data[:, 4], raw_data[:, 6])
    # draw_scatter_plot(raw_data[:, 4], raw_data[:, 7])
    # draw_scatter_plot(raw_data[:, 4], raw_data[:, 8])
    # draw_scatter_plot(raw_data[:, 4], raw_data[:, 9])

    # draw_one_other_scatter_plot(raw_data[:, 2], raw_data[:, 5:])
    # draw_one_other_scatter_plot(raw_data[:, 1], raw_data[:, 5:])
    # draw_one_other_scatter_plot(raw_data[:, 2], raw_data[:, 5:])
    # draw_one_other_scatter_plot(raw_data[:, 3], raw_data[:, 5:])
    # draw_one_other_scatter_plot(raw_data[:, 4], raw_data[:, 5:])

    # draw_all_scatter_plot(raw_data)