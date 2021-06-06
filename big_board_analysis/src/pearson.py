# -*- coding: utf-8 -*-
__author__ = 'Kay'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import random


# import seaborn as sns
# sns.set(style="darkgrid", palette="Set1")

def my_pearson(in_name, start, end):
    data = np.loadtxt(in_name, dtype=float)
    for i in range(start, end):
        for j in range(i+1, end+1):
            print(i, j)
            # print(data[:, i])
            # print(data[:, j])
            print(pearsonr(data[:, i], data[:, j]))


def write_pearson(input_file, output_dir):
    '''
    根据输入文件，写入皮尔逊相关性
    :param input_file:
    :return:
    '''
    raw_data = np.loadtxt(input_file, 'float')
    for goal in range(0, 5):
        goal_file = open('%s/%s.txt' % (output_dir, goal), 'a')
        goal_col = raw_data[:, goal]
        for emotion in range(0, 5):
            emotion_row = []
            for lags in range(1, 6):
                emotion_col = raw_data[:, (emotion + 1) * 5 + lags - 1]

                pea, x = pearsonr(goal_col, emotion_col)
                print(pea, x)
                emotion_row.append(str(pea))
                # print(goal, emotion, lags, pea)
            goal_file.write(' '.join(emotion_row) + '\n')


def wwwj_perason(in_name):
    data = pd.read_csv(in_name)
    max_lag = 30
    x = range(1, max_lag + 1)
    for index, goal in enumerate(['close', 'open', 'high', 'low', 'volume']):
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        for mood_color, mood in enumerate(['anger', 'disgust', 'joy', 'sadness', 'fear']):
            y = []
            for lag in range(1, max_lag + 1):
                a = data.ix[lag:, goal]
                b = data.ix[:len(data) - 1 - lag, mood]
                # a = a.sample(frac=1)
                # b = b.sample(frac=1)
                # print(len(a), len(b))
                pea, p_value = pearsonr(a, b)
                y.append(pea)

                # print(goal, mood, 'lag:', lag, 'rho: %.2f' % pea, 'p-value: %.4f' % p_value)

                # if p_value < 0.05 and pea > 0.2 and lag <= 5:
                # if p_value < 0.05 and pea > 0.2:
                #     print(goal, mood, 'lag:', lag, 'rho: %.2f' % pea, 'p-value: %.4f' % p_value)
            # print(x, y)


            # ax.plot(x, y, label=mood)
            # ax.plot(x, pd.rolling_mean(pd.Series(y), 5), label=mood)

            # 每5天聚合一次
            x = [i for i in x if i % 5 == 0]

            bingo = False
            for i in range(5):
                if abs(y[i]) > 0.2:
                    bingo = True

            mean_y = []
            for i in x:
                temp = 0
                for j in range(1, 6):
                    temp += y[i - j]
                mean_y.append(temp / 5)
            y = mean_y

            # 只要前5日达到0.2之上
            if bingo:
                ax.plot(x, y, 'o-', label=mood)
                # ax.plot(x, y, 'o-', label=mood, color=sns.color_palette('Set1')[mood_color])

        plt.yticks(fontsize=12)
        # plt.xticks(fontsize=15)
        plt.xticks([5, 10, 15, 20, 25, 30], ['1-5', '6-10', '11-15', '16-20', '21-25', '26-30'],
                   fontsize=12)
        # plt.tight_layout()
        plt.grid(True)
        plt.legend(loc=1)
        plt.ylabel('$Correlation\ coefficient$', fontsize=12)
        plt.xlabel('$Days\ in\ advance$', fontsize=12)
        plt.xlim(3, max_lag + 2)
        plt.ylim(-0.6, 0.6)
        # plt.show()
        # plt.savefig('../tmp/pearson_' + goal + '.pdf')
        plt.savefig('../tmp/pearson_' + str(index) + '.pdf', dpi=300)
        # plt.close()


def wwwj_perason_bootstrap(in_name):
    data = pd.read_csv(in_name)
    max_lag = 30
    x = range(1, max_lag + 1)
    for goal in ['close', 'open', 'high', 'low', 'volume']:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for mood in ['anger', 'disgust', 'joy', 'sadness', 'fear']:
            y = []
            for lag in range(1, max_lag + 1):
                mean = []

                a = data.ix[lag:, goal]
                b = data.ix[:len(data) - 1 - lag, mood]
                # a = a.sample(frac=1)
                # b = b.sample(frac=1)
                # print(len(a), len(b))
                for _ in range(100):
                    bingo_a = random.sample(range(len(a)), 150)
                    bingo_b = random.sample(range(len(b)), 150)
                    pea, p_value = pearsonr(a.iloc[bingo_a], b.iloc[bingo_b])
                    mean.append(pea)
                # print(goal, mood, 'lag:', lag, 'rho: %.2f' % pea, 'p-value: %.4f' % p_value)
                mean = np.array(mean).mean()
                print(mean)
                y.append(mean)
                # if p_value < 0.05 and pea > 0.2 and lag <= 5:
                # if p_value < 0.05 and pea > 0.2:
                #     print(goal, mood, 'lag:', lag, 'rho: %.2f' % pea, 'p-value: %.4f' % p_value)
            # print(x, y)
            ax.plot(x[5:], pd.rolling_mean(y, 5), label=mood)
            plt.yticks(fontsize=15)
            plt.xticks(fontsize=15)
            plt.tight_layout()
            plt.grid(True)
            plt.legend(loc=1)
            plt.ylabel('$Correlation\ coefficient$', fontsize=12)
            plt.xlabel('$Days\ in\ advance$', fontsize=12)
            plt.xlim(1, max_lag)
            plt.ylim(-0.5, 0.5)

        # plt.show()
        plt.savefig('../tmp/bootstrap_pearson_' + goal + '.pdf', dpi=300)
        plt.savefig('../tmp/shuffle_bootstrap_pearson_' + goal + '.pdf', dpi=300)
        plt.close()


def shuffle_pearson_error_bar(in_name):
    '''
    画出shuffle后的error_bar图作为对比
    :param in_name: 
    :return: 
    '''
    data = pd.read_csv(in_name)
    max_lag = 5

    for goal in ['close', 'open', 'high', 'low', 'volume']:
        out_file = open('../data/proportion/shuffle_error_bar/shuffle_error_bar_{}.txt'.format(goal), 'w')
        out_file.write('goal,emotion,lags,mean,max,min\n')
        for mood in ['anger', 'disgust', 'joy', 'sadness', 'fear']:
            for lag in range(1, max_lag + 1):
                y = []
                a = data.ix[lag:, goal]
                b = data.ix[:len(data) - 1 - lag, mood]

                for _ in range(100):
                    # 随机sample150个
                    a = a.sample(n=150)
                    b = b.sample(n=150)
                    # a = a.sample(n=150, random_state=43)
                    # b = b.sample(n=150, random_state=29)
                    pea, p_value = pearsonr(a, b)
                    y.append(pea)
                y = np.array(y)
                mean = y.mean()
                std = y.std()
                print(mean, mean + std, mean - std)
                out_file.write(','.join([goal, mood, str(lag), str(mean), str(mean + std), str(mean - std)]) + '\n')


'''
计算皮尔逊相关性并写入文件
'''

if __name__ == '__main__':
    # write_pearson('../data/proportion/shuffle.txt',
    #               '../data/proportion/shuffle_pearson')

    # my_pearson('/Users/Kay/Project/ML/big_board_analysis/data/2015年_涨跌幅_09_15_mood.txt', 1, 6)

    # wwwj_perason_bootstrap('../data/20141201-20150916_pro.txt')
    wwwj_perason('../data/20141201-20150916_pro.txt')

    # shuffle_pearson_error_bar('../data/20141201-20150916_pro.txt')
