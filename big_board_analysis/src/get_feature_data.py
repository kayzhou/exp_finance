__author__ = 'Kay Zhou'

'''
不同用户的情绪来试一下，直接用k-means已经做好的结果来做，组装好数据后进行normalization.
'''

import pandas as pd


def get_regress_data(in_name, out_name):
    # 三分类结果
    target_data = pd.read_csv('../data/20141201-20150916_pro.txt')
    # print(target_data)
    data = pd.read_csv(in_name).drop('date', axis=1)
    # print(data)

    out_file = open(out_name, 'w')
    for i, row in target_data.iterrows():
        new_row = []
        # print(i, row)
        targets = list(row[:5])
        targets = [str(float(t)) for t in targets]
        new_row.extend(targets)

        features = list(data.ix[i])
        features = [str(round(f, 5)) for f in features]
        new_row.extend(features)
        out_file.write(','.join(new_row) + '\n')


def get_train_data(in_name, out_name):
    # 三分类结果
    target_data = pd.read_csv('../data/proportion/20141201-20150916_比例_二分类_true.txt', header=None, delimiter=' ')
    # print(target_data)
    data = pd.read_csv(in_name).drop('date', axis=1)
    # print(data)

    out_file = open(out_name, 'w')
    for i, row in target_data.iterrows():
        new_row = []
        # print(i, row)
        targets = list(row[:5])
        targets = [str(int(t)) for t in targets]
        new_row.extend(targets)
        for j in range(5):
            features = list(data.ix[i + j])
            features = [str(round(f, 5)) for f in features]
            new_row.extend(features)
        out_file.write(' '.join(new_row) + '\n')


def get_test_data(in_name, out_name):
    # 三分类结果
    # target_data = pd.read_csv('../data/proportion/测试数据/20150917-20151207_比例_三分类_true.txt', header=None, delimiter=' ')
    target_data = pd.read_csv('../data/proportion/测试数据/20150917_20151207_比例_二分类_true.txt', header=None, delimiter=' ')
    # print(target_data)
    data = pd.read_csv(in_name)

    # 测试数据所需的情绪数据从9月10日开始
    data = data[data.date >= '2015-09-10']
    data = data.drop('date', axis=1)
    # print(data)

    out_file = open(out_name, 'w')
    for i, row in target_data.iterrows():
        new_row = []
        # print(i, row)
        targets = list(row[:5])
        targets = [str(int(t)) for t in targets]
        new_row.extend(targets)
        for j in range(5):
            features = list(data.ix[i + j + 191])
            features = [str(round(f, 5)) for f in features]
            new_row.extend(features)
        out_file.write(' '.join(new_row) + '\n')


if __name__ == '__main__':
    # get_regress_data('../data/proportion/emotion_foll_level_0_pro.csv', '../data/proportion/训练数据/user_0_regression.txt')
    # get_regress_data('../data/proportion/emotion_foll_level_1_pro.csv', '../data/proportion/训练数据/user_1_regression.txt')
    # get_regress_data('../data/proportion/emotion_foll_level_2_pro.csv', '../data/proportion/训练数据/user_2_regression.txt')

    get_train_data('../data/proportion/emotion_foll_level_0_pro.csv', '../data/proportion/训练数据/user_0_二分类.txt')
    get_test_data('../data/proportion/emotion_foll_level_0_pro.csv', '../data/proportion/测试数据/user_0_二分类.txt')
    get_train_data('../data/proportion/emotion_foll_level_1_pro.csv', '../data/proportion/训练数据/user_1_二分类.txt')
    get_test_data('../data/proportion/emotion_foll_level_1_pro.csv', '../data/proportion/测试数据/user_1_二分类.txt')
    get_train_data('../data/proportion/emotion_foll_level_2_pro.csv', '../data/proportion/训练数据/user_2_二分类.txt')
    get_test_data('../data/proportion/emotion_foll_level_2_pro.csv', '../data/proportion/测试数据/user_2_二分类.txt')
