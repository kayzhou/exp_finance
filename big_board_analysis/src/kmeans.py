# -*- coding: utf-8 -*-
__author__ = 'Kay'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from collections import Counter


def kmeans_cluster_centers(X, n=3):
    X = [[x] for x in X]
    est = KMeans(n_clusters=n, init='random')
    est.fit(X)
    print(np.bincount(est.labels_))
    return est.cluster_centers_


def kmeans(X, n=3):
    # cs = kmeans_cluster_centers(X, n).tolist(); cs.sort()
    # print('聚类中心 =', cs)
    est = KMeans(n_clusters=n, init='random')
    X = [[x] for x in X]
    est.fit(X)
    re = est.labels_
    # print('聚类结果 =', re)
    return re


def regress_to_classify(in_name, out_name):
    '''
    回归问题转分类问题
    :param in_name:
    :param out_name:
    :return:
    '''

    def kmeans_classify(X, X_test, n=3):
        clusters = kmeans_cluster_centers(X, n).tolist()
        clusters.sort()
        print('聚类中心 =', clusters)

        # 计算分割点
        # print(len(clusters))
        points = [0] * (len(clusters) - 1)
        for i in range(len(points)):
            points[i] = (clusters[i][0] + clusters[i + 1][0]) / 2
        print('分割:', points)

        re = []
        dist = [0] * n
        for x in X_test:
            for i in range(n):
                dist[i] = abs(x - clusters[i][0])
            label = np.argmin(dist)
            # print(x, label)
            re.append(label)
        # print(re)
        return re


    txt = np.loadtxt(in_name, 'float')
    test_txt = np.loadtxt(in_name, 'float')
    new_txt = list()

    # 参数
    clusters_count = 3

    # 文本前五列，三分类
    for i in range(5):
        new_col = kmeans_classify(txt[:, i], test_txt[:, i], n=clusters_count)
        tmp = np.bincount(np.array(new_col))
        print('类中个数:', tmp, '\n')
        new_txt.append(new_col)

    for i in range(5, 30):
        new_col = test_txt[:, i]
        new_txt.append(new_col)

    new_txt = np.array(new_txt).transpose()
    np.savetxt(out_name, new_txt, fmt='%.4f')


def regress_to_2_classify(in_name, out_name):
    '''
    回归问题转二分类问题
    :param in_name:
    :param out_name:
    :return:
    '''
    def convert_to_sign(x):
        new_x = []
        print(x)
        for i in x:
            # print(x)
            if i > 0:
                new_x.append(1)
            else:
                new_x.append(-1)
        return new_x


    txt = np.loadtxt(in_name, 'float')
    new_txt = list()

    # 文本前五列，三分类
    for i in range(0, 5):
        new_col = convert_to_sign(txt[:, i])
        # print(new_col)

        print('类中个数:', Counter(new_col), '\n')
        new_txt.append(new_col)

    for i in range(5, 30):
        new_col = txt[:, i]
        new_txt.append(new_col)

    new_txt = np.array(new_txt).transpose()
    np.savetxt(out_name, new_txt, fmt='%.4f')


def regress_to_classify_test(in_name, in_name_test, out_name):
    '''
    回归问题转分类问题，利用已有的数据对测试数据（test）做离散化
    '''

    def kmeans_classify(X, X_test, n=3):
        clusters = kmeans_cluster_centers(X, n).tolist()
        clusters.sort()
        print('聚类中心 =', clusters)

        # 计算分割点
        # print(len(clusters))
        points = [0] * (len(clusters) - 1)
        for i in range(len(points)):
            points[i] = (clusters[i][0] + clusters[i + 1][0]) / 2
        print('分割:', points)

        re = []
        dist = [0] * n
        for x in X_test:
            for i in range(n):
                dist[i] = abs(x - clusters[i][0])
            label = np.argmin(dist)
            # print(x, label)
            re.append(label)
        # print(re)
        return re


    txt = np.loadtxt(in_name, 'float')
    test_txt = np.loadtxt(in_name_test, 'float')
    new_txt = list()
    # 参数
    clusters_count = 3

    # 文本前五列，三分类
    for i in range(5):
        new_col = kmeans_classify(txt[:, i], test_txt[:, i], n=clusters_count)
        tmp = np.bincount(np.array(new_col))
        print('类中个数:', tmp, '\n')
        new_txt.append(new_col)

    for i in range(5, 30):
        new_col = test_txt[:, i]
        new_txt.append(new_col)

    new_txt = np.array(new_txt).transpose()
    np.savetxt(out_name, new_txt, fmt='%.4f')


def regress_to_classify_csv(in_name, out_name):
    '''
    回归问题转分类问题, 搞定close
    :param in_name:
    :param out_name:
    :param count_of_class:
    :return:
    '''
    data = pd.read_csv(in_name)
    re = pd.Series(kmeans(data.close))
    df = pd.concat([data, re], axis=1)
    df.to_csv(out_name)


def label_to_pro(in_name):
    '''
    探索三种label下的情绪比例
    :param in_name:
    :return:
    '''
    data = pd.read_csv('temp.csv')
    pro = [[0] * 5, [0] * 5, [0] * 5]
    label_count = [0.0, 0.0, 0.0]

    for i in np.arange(data.shape[0]):
        # print(i)
        m_0 = float(data['anger'][i])
        m_1 = float(data['disgust'][i])
        m_2 = float(data['joy'][i])
        m_3 = float(data['sadness'][i])
        m_4 = float(data['fear'][i])
        _sum = m_0 + m_1 + m_2 + m_3 + m_4

        label = data['label'][i]
        label_count[label] += 1

        print(label, m_0 / _sum, m_1 / _sum, m_2 / _sum, m_3 / _sum, m_4 / _sum)

        pro[label][0] += (m_0 / _sum)
        pro[label][1] += (m_1 / _sum)
        pro[label][2] += (m_2 / _sum)
        pro[label][3] += (m_3 / _sum)
        pro[label][4] += (m_4 / _sum)

    print(pro)
    for i in np.arange(3):
        for j in np.arange(5):
            pro[i][j] = pro[i][j] / label_count[i]

    print(label_count)
    print(pro)


def kmeans_matrix_zhao(num):
    '''
    给赵老师做的K-means实验
    :param num:
    :return:
    '''
    X = []
    for line in open('../data/26-29NMI/' + num + 'NMI.csv'):
        if line.strip():
            X.append([float(word) for word in line.strip().split(',') if word])

    for i in list(range(1000)):
        print(i)
        fi1 = open('../data/kmeans_result/' + num + 'NMI_label_' + str(i) + '.txt', 'w')
        fi2 = open('../data/kmeans_result/' + num + 'NMI_cluster_centers_' + str(i) + '.txt', 'w')
        kmeans = KMeans(n_clusters=5, init='random')
        kmeans.fit(X)

        for k in kmeans.labels_:
            fi1.write(str(k) + '\n')
        for k in kmeans.cluster_centers_:
            fi2.write(str(k) + '\n')
        fi1.close()
        fi2.close()


if __name__ == '__main__':
    # regress_to_classify('../data/proportion/20141201-20150916_pro.txt',
    #                     '../data/proportion/20141201-20150916_比例_三分类.txt')
    #
    # regress_to_classify_test('../data/proportion/20141201-20150916_pro.txt',
    #                          '../data/proportion/测试数据/20150917_20151207.txt',
    #                          '../data/proportion/测试数据/20150917-20151207_比例_三分类.txt')

    # regress_to_2_classify('../data/proportion/20141201-20150916_pro.txt',
    #                       '../data/proportion/20141201-20150916_比例_二分类.txt')
    regress_to_2_classify('../data/proportion/测试数据/20150917_20151207.txt',
                          '../data/proportion/测试数据/20150917_20151207_比例_二分类.txt')


    # regress_to_classify_test('../data/proportion/20141201-20150916_pro.txt',
    #                          '../data/proportion/测试数据/20150917_20151207.txt',
    #                          '../data/proportion/测试数据/20150917-20151207_比例_七分类.txt')

    # in_name = '/Users/Kay/Project/EXP/stock_market_analysis/get_train_big_board/VIP/20141201-20160501_day.csv'
    # regress_to_classify_csv(in_name, 'temp.csv')
    # label_to_pro(in_name)

