# -*- coding: utf-8 -*-
__author__ = 'Kay'


import numpy as np
from sklearn import linear_model


def get_stock_features(in_name, max_len=5):

    # 数据准备
    text_lines = open(in_name).readlines()[1:]
    # print('Format:', text_lines[0])

    sources_all = []
    targets_all = []
    dim = 8  # 每日指标数量

    for line in text_lines:
        d = line.strip().split(',')
        '''
        Example:
        date,open,high,low,close,volume,amount,bamt,samt
        20160704,21.99,21.99,21.99,21.99,426.63,9381.59,0,9381.59
        '''
        S = []
        extra = [] # 额外补充信息，不参与建模，用于模型的评估
        # 六个标准指标
        S.append(float(d[1]))
        S.append(float(d[2]))
        S.append(float(d[3]))
        S.append(float(d[4]))
        S.append(float(d[5]) / 10e3)
        S.append(float(d[6]) / 10e3)

        # 资金流入流出
        S.append(float(d[7]) / 10e3)
        S.append(float(d[8]) / 10e3)

        if len(sources_all) == 0: # 初始化，因为第一天并不知道变化
            S.append(0.00001)
            extra.append(0.00001)
            extra.append(0.00001)

        else:
            # 前一天信息
            last = sources_all[-1]
            last_close = last[3]

            # 添加变化率
            # 收盘价对于昨日的变化率
            S.append((S[3] - last_close) / last_close)
            extra.append((S[1] - last_close) / last_close)
            extra.append((S[2] - last_close) / last_close)

        sources_all.append(S)
        # print(str(extra[0]) + ',' + str(extra[1]))

        # S[-1] -> 涨跌幅
        T = S[-1]
        # T = 1 if S[-1] > 0 else 0
        targets_all.append(T)

    print("数据大小:", len(sources_all))

    # Cut the text in semi-redundant sequences of max-len characters
    step = 1

    sentences = []
    next_chars = []
    for i in list(range(0, len(sources_all) - max_len, step)):
        sentences.append(sources_all[i: i + max_len])
        next_chars.append(targets_all[i + max_len])


    # 向量化
    X = np.zeros((len(sentences), max_len, dim), dtype=np.float32)
    y = np.zeros((len(sentences), 1), dtype=np.float32)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            for g in np.arange(0, dim):
                X[i, t, g] = char[g]
        y[i, 0] = next_chars[i]


    train_test = 3 / 5
    train_size = int(len(X) * train_test)
    print('训练数据比例:', train_test)
    print('训练数据:', train_size)

    X_train = X[:train_size]
    y_train = y[:train_size]

    X_test = X[train_size:]
    y_test = y[train_size:]

    return X_train, y_train, X_test, y_test


def main():
    X, y, X_test, y_test = get_stock_features('../data/data-with-BS.txt', max_len=13)
    # print(X, y, X_test, y_test)

    # 训练数据和测试数据的写入
    out_file = open('train.txt', 'w')
    for count, x in enumerate(X):
        line = []
        for i in x:
            for j in i:
                line.append(str(j))
        out_file.write(','.join(line) + ',' + str(y[count][0]) + '\n')
    out_file.close()

    out_file = open('test.txt', 'w')
    for count, x in enumerate(X_test):
        line = []
        for i in x:
            for j in i:
                line.append(str(j))
        out_file.write(','.join(line) + ',' + str(y_test[count][0]) + '\n')
    out_file.close()


if __name__ == '__main__':
    main()