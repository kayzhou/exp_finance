#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import random
import sklearn.preprocessing as sk_pre
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import sgd
from keras.layers import LSTM
from keras.utils import np_utils
from keras.layers import GRU
from keras.utils.data_utils import get_file


def s_gru(in_name):

    # 数据准备
    # text_lines = open('000001.csv').readlines()[1:]
    # print('Format:', text_lines[0])
    #
    # sources_all = []
    # targets_all = []
    # dim = 6
    #
    # for line in reversed(text_lines):
    #     lw = line.split(',')
    #     # lw[3] -> close, lw[5] -> volume, lw[6] -> amount
    #     S = [float(lw[3]) / 1000, float(lw[5]) / 10e10, float(lw[6]) / 10e11]
    #
    #     # init
    #     if len(sources_all) == 0:
    #         S.append(0.00001)
    #         S.append(0.00001)
    #         S.append(0.00001)
    #
    #     else:
    #         # 前一天信息
    #         last = sources_all[-1]
    #         # 添加变化率
    #         S.append((S[0] - last[0]) / last[0])
    #         S.append((S[1] - last[1]) / last[1])
    #         S.append((S[2] - last[2]) / last[2])
    #
    #     sources_all.append(S)
    #
    #     # S[3] -> 收益率
    #     # T = S[3]
    #     T = 1 if S[3] > 0 else 0
    #     targets_all.append(T)


    # 数据准备
    text_lines = open(in_name).readlines()[1:][::-1]
    # print('Format:', text_lines[0])

    sources_all = []
    targets_all = []
    dim = 16 # 每日指标数量

    for line in reversed(text_lines):
        d = line.strip().split(',')
        '''
        Example:
        date,open,high,close,low,volume,price_change,p_change,ma5,ma10,ma20,v_ma5,v_ma10,v_ma20
        2016-07-01,2931.8,2944.99,2932.48,2925.81,1412462.25,2.87,0.1,2920.388,2902.111,2900.136,1638990.53,1577995.81,1564399.24
        '''
        S = []
        # 五个标准指标
        S.append(float(d[1]) / 10e5)
        S.append(float(d[2]) / 10e5)
        S.append(float(d[3]) / 10e5)
        S.append(float(d[4]) / 10e5)
        S.append(float(d[5]) / 10e7)

        # ma5~20, 股价均线
        S.append(float(d[8]) / 10e5)
        S.append(float(d[9]) / 10e5)
        S.append(float(d[10]) / 10e5)

        # v_ma5~20, 成交量均线
        S.append(float(d[11]) / 10e7)
        S.append(float(d[12]) / 10e7)
        S.append(float(d[13]) / 10e7)

        # init
        if len(sources_all) == 0:
            S.append(0.00001)
            S.append(0.00001)
            S.append(0.00001)
            S.append(0.00001)
            S.append(0.00001)

        else:
            # 前一天信息
            last = sources_all[-1]
            last_close = last[2]

            # 添加变化率
            # 五个指标相对于昨日的变化率
            S.append((S[0] - last_close) / last_close)
            S.append((S[1] - last_close) / last_close)
            S.append((S[2] - last_close) / last_close)
            S.append((S[3] - last_close) / last_close)
            S.append((S[4] - last[4]) / last[4])

        sources_all.append(S)

        # S[11] -> 收益率
        # T = S[11]
        T = 1 if S[11] > 0 else 0
        targets_all.append(T)


    print("数据大小:", len(sources_all))
    # print(len(targets_all))

    train_test = 4 / 5
    train_size = int(len(sources_all) * train_test)
    print('训练数据比例:', train_test)
    print(sources_all)
    print(targets_all)
    targets_all = np_utils.to_categorical(targets_all, 2)

    sources = sources_all[:train_size]
    targets = targets_all[:train_size]
    sources_test = sources_all[train_size:]
    targets_test = targets_all[train_size:]


    # Cut the text in semi-redundant sequences of maxlen characters
    maxlen = 10 # 前10交易日的数据去做预测
    step = 1

    sentences = []
    next_chars = []
    for i in list(range(0, len(sources) - maxlen, step)):
        sentences.append(sources[i: i + maxlen])
        next_chars.append(targets[i + maxlen])
    print('训练数据:', len(sentences))


    sentences_test = []
    next_chars_test = []
    for i in list(range(0, len(sources_test) - maxlen, step)):
        sentences_test.append(sources_test[i: i + maxlen])
        next_chars_test.append(targets_test[i + maxlen])
    print('测试数据:', len(sentences_test))

    # 搞成三维数据
    print('向量化 ...')
    X = np.zeros((len(sentences), maxlen, dim), dtype=np.float32)
    y = np.zeros((len(sentences), 1), dtype=np.float32)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            for g in np.arange(0, dim):
                X[i, t, g] = char[g]
        y[i, 0] = next_chars[i]


    X_test = np.zeros((len(sentences_test), maxlen, dim), dtype=np.float32)
    y_test = np.zeros((len(sentences_test), 1), dtype=np.float32)
    for i, sentence in enumerate(sentences_test):
        for t, char in enumerate(sentence):
            for g in np.arange(0, dim):
                X_test[i, t, g] = char[g]
        y_test[i, 0] = next_chars_test[i]


    # Build the model
    print('模型构建中 ...')

    model = Sequential()
    model.add(LSTM(128, input_shape=(maxlen, dim)))
    model.add(Dropout(0.2))
    # model.add(GRU(1024, input_shape=(1024, 3))) # ???
    # model.add(Dropout(0.2))
    model.add(Dense(1))
    opt = sgd(lr=0.001)
    model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    # Train the model, output generated text after each iteration
    for iteration in list(range(1, 6)):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        model.fit(X, y, batch_size=128, nb_epoch=1)
        print("在训练集上评估:", model.evaluate(X, y))
        print("在测试集上评估:", model.evaluate(X_test, y_test))


    print("done!")


if __name__ == '__main__':
    # s_gru(sys.argv[1])
    s_gru('../data/stock-data/sh.txt')
