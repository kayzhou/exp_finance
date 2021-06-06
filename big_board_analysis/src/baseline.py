__author__ = 'Kay Zhou'

'''
使用基本数据对股市进行预测，用作对比实验
'''

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def get_features_file(in_name='../data/sh.csv'):
    data = pd.read_csv(in_name)
    d = data.p_change

    # 用提前5天去预测
    max_len = 5

    features = {}
    for lag in range(1, max_len + 1):
        features['lag' + str(lag)] = list(d[max_len - lag: -lag])
    features['date'] = list(data.date[max_len:])

    # -- 挑选目标 --
    last_close = data.close - data.price_change
    target = (data.close - last_close) / last_close * 100

    # target = data.volume

    print(target)

    features['target'] = list(target[max_len:])

    features_data = pd.DataFrame(features).set_index('date')
    features_data[features_data.index < '2015-09-17'].to_csv('../data/baseline_train.csv')
    features_data[features_data.index >= '2015-09-17'].to_csv('../data/baseline_test.csv')


def to_classify(train_name, test_name, n):
    train_data = pd.read_csv(train_name)
    test_data = pd.read_csv(test_name)

    def kmeans(X, n=3):

        def kmeans_cluster_centers(X, n=3):
            X = [[x] for x in X]
            est = KMeans(n_clusters=n, init='random')
            est.fit(X)
            print(np.bincount(est.labels_))
            return est.cluster_centers_

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
        for x in X:
            for i in range(n):
                dist[i] = abs(x - clusters[i][0])
            label = np.argmin(dist)
            # print(x, label)
            re.append(label)

        print('聚类结果 =', re)
        return re

    def kmeans_classify(X, X_test, n=3):

        def kmeans_cluster_centers(X, n=3):
            X = [[x] for x in X]
            est = KMeans(n_clusters=n, init='random')
            est.fit(X)
            print(np.bincount(est.labels_))
            return est.cluster_centers_

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

        print('聚类结果 =', re)
        return re

    if n == 2:
        temp = []
        for i in train_data['target']:
            if i >= 0:
                temp.append(1)
            else:
                temp.append(-1)
        train_data['class'] = temp

        temp = []
        for i in test_data['target']:
            # print(i)
            if i >= 0:
                temp.append(1)
            else:
                temp.append(-1)
        test_data['class'] = temp

    else:
        train_data['class'] = kmeans(train_data.target, n=n)
        test_data['class'] = kmeans_classify(train_data.target, test_data.target, n=n)

    max_len = 5

    for lag in range(1, max_len + 1):
        scaler = MinMaxScaler()
        x = train_data['lag' + str(lag)].values.reshape(-1, 1)
        train_data['lag' + str(lag)] = scaler.fit_transform(X=x)
        x_test = scaler.transform(test_data['lag' + str(lag)].values.reshape(-1, 1))
        test_data['lag' + str(lag)] = x_test

    train_data.set_index('date').to_csv(train_name)
    test_data.set_index('date').to_csv(test_name)


def train_test(train_name, test_name):
    train_data = pd.read_csv(train_name)
    test_data = pd.read_csv(test_name)

    def get_X_y(data):
        max_len = 5
        X = []
        y = []
        for i, row in data.iterrows():
            temp = []
            for lag in range(1, max_len + 1):
                temp.append(row['lag' + str(lag)])
            X.append(temp)
            y.append(row['class'])
        # print(X)
        # print(y)
        return X, y

    clf = RandomForestClassifier()
    # clf = SVC(C=1000, gamma=10)
    X_train, y_train = get_X_y(train_data)
    X_test, y_test = get_X_y(test_data)
    clf.fit(X_train, y_train)
    print(clf.predict(X_test))
    print('准确率 =', clf.score(X_test, y_test))


if __name__ == '__main__':
    get_features_file()
    to_classify('../data/baseline_train.csv', '../data/baseline_test.csv', n=3)
    train_test('../data/baseline_train.csv', '../data/baseline_test.csv')
