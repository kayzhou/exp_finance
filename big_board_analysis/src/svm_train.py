# -*- coding: utf-8 -*-
__author__ = 'Kay'


import numpy as np
from sklearn.svm import SVC
from sklearn.svm.libsvm import cross_validation
from sklearn.datasets import load_svmlight_file
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, f1_score

'''
SVM K-means 交叉验证结果：
32768.0 0.00048828125 58.1152
0.03125 0.0078125 62.8272
8.0 0.5 57.0681
8.0 0.5 63.3508
8.0 8.0 67.0157

2015-10-28

调参结果：

三分类：
0 32768.0 0.00048828125 58.1152
1 0.03125 0.0078125 67.0157 pity
2 0.03125 0.0078125 52.356 pity
3 512.0 0.03125 61.2565
4 8.0 2.0 68.0628

二分类：
0 2048.0 0.00048828125 61.2565
1 32.0 0.5 65.9686

2015-11-13

调参结果

    param = '-t 2 -c 32768.0 -g 0.00048828125' # 3-close √
    param = '-t 2 -c 512.0 -g 0.0078125' # 3-open √
    param = '-t 2 -c 2048.0 -g 0.00048828125' # 3-high √
    param = '-t 2 -c 8.0 -g 0.5' # 3-low √
    param = '-t 2 -c 8.0 -g 8.0' # 3-volume √

    param = '-t 2 -c 2048.0 -g 0.00048828125' # 2-close
    param = '-t 2 -c 32.0 -g 0.5' # 2-open

'''

# X, y = load_svmlight_file('../data/proportion/SVM_DATA/class3_GOAL0.txt')
# clf = SVC(C=32768.0, gamma=0.00048828125, probability=True)

# X, y = load_svmlight_file('../data/proportion/SVM_DATA/class3_GOAL1.txt')
# clf = SVC(C=512.0, gamma=0.0078125, probability=True)

# X, y = load_svmlight_file('../data/proportion/SVM_DATA/class3_GOAL2.txt')
# clf = SVC(C=2048, gamma=0.00048828125, probability=True)

# X, y = load_svmlight_file('../data/proportion/SVM_DATA/class3_GOAL3.txt')
# clf = SVC(C=8.0, gamma=0.5, probability=True)

# X, y = load_svmlight_file('../data/proportion/SVM_DATA/class3_GOAL4.txt')
# clf = SVC(C=8 , gamma=8, probability=True)

# X, y = load_svmlight_file('../data/proportion/SVM_DATA/class2_GOAL0.txt')
# clf = SVC(C=2048, gamma=0.00048828125, probability=True)

# X, y = load_svmlight_file('../data/proportion/SVM_DATA/class2_GOAL1.txt')
# clf = SVC(C=32, gamma=0.5, probability=True)

# EMOTION-SELECTED

# X, y = load_svmlight_file('../data/proportion/SVM_DATA/selected_class3_GOAL0.txt')
# clf = SVC(C=32768.0, gamma=2, probability=True)

# X, y = load_svmlight_file('../data/proportion/SVM_DATA/selected_class3_GOAL1.txt')
# clf = SVC(C=2, gamma=8, probability=True)

# X, y = load_svmlight_file('../data/proportion/SVM_DATA/selected_class3_GOAL2.txt')
# clf = SVC(C=512, gamma=0.0078125, probability=True)

# X, y = load_svmlight_file('../data/proportion/SVM_DATA/selected_class3_GOAL3.txt')
# clf = SVC(C=8.0, gamma=2, probability=True)

# X, y = load_svmlight_file('../data/proportion/SVM_DATA/selected_class3_GOAL4_sadness_fear.txt')
# clf = SVC(C=8 , gamma=8, probability=True)

# X, y = load_svmlight_file('../data/proportion/SVM_DATA/selected_class2_GOAL0.txt')
# clf = SVC(C=128, gamma=0.5, probability=True)

# X, y = load_svmlight_file('../data/proportion/SVM_DATA/selected_class2_GOAL1.txt')
# clf = SVC(C=8, gamma=2, probability=True)


if __name__ == '__main__':
    # 五分类
    # params = [
    #     {'C': 2.0, 'gamma': 0.25},
    #     {'C': 2.0, 'gamma': 0.125},
    #     {'C': 4.0, 'gamma': 0.25},
    #     {'C': 16, 'gamma': 0.125},
    #     {'C': 2.0, 'gamma': 2.0},
    # ]

    # 七分类
    # params = [
    #     {'C': 2, 'gamma': 1.0},
    #     {'C': 8, 'gamma': 0.3125},
    #     {'C': 2, 'gamma': 2},
    #     {'C': 0.625, 'gamma': 32},
    #     {'C': 2.0, 'gamma': 0.5},
    # ]

    # User 0 - 三分类
    params = [
        {'C': 4, 'gamma': 1},
        {'C': 1, 'gamma': 1},
        {'C': 2, 'gamma': 0.5},
        {'C': 2, 'gamma': 0.5},
        {'C': 1, 'gamma': 0.5},
    ]

    # User 1 - 三分类
    # params = [
    #     {'C': 4, 'gamma': 1},
    #     {'C': 2, 'gamma': 1},
    #     {'C': 2, 'gamma': 0.5},
    #     {'C': 2, 'gamma': 0.5},
    #     {'C': 1, 'gamma': 1},
    # ]

    # User 2 - 三分类
    # params = [
    #     {'C': 1, 'gamma': 1},
    #     {'C': 0.0009765625, 'gamma': 0.0009765625},
    #     {'C': 0.5, 'gamma': 16},
    #     {'C': 0.0009765625, 'gamma': 0.0009765625},
    #     {'C': 1, 'gamma': 1},
    # ]


    # 读取数据
    train_data = np.loadtxt('../data/proportion/训练数据/20141201-20150916_比例_三分类_true.txt')
    test_data = np.loadtxt('../data/proportion/测试数据/20150917-20151207_比例_三分类_true.txt')

    # train_data = np.loadtxt('../data/proportion/训练数据/user_0_true.csv')
    # test_data = np.loadtxt('../data/proportion/测试数据/user_0_true.csv')

    # raw_data = np.loadtxt('../data/proportion/20141201-20150916_比例_五分类_true.txt')
    # raw_data = np.loadtxt('../data/proportion/20141201-20150916_比例_七分类_true.txt')
    # raw_data = np.loadtxt('../data/proportion/classify/kmeans_3_class.txt')

    X_train = train_data[:, 5:]
    X_test = test_data[:, 5:]

    for i in range(5):
        print('-' * 20, i, '-' * 20)
        p = params[i]
        y_train = train_data[:, i]
        y_test = test_data[:, i]

        clf = SVC(C=p['C'], gamma=p['gamma'])
        clf.fit(X_train, y_train)
        y_hat = clf.predict(X_train)
        print('train >>>')
        print('预测结果:', y_hat)
        print('实际结果:', y_train)
        print('准确率: %.2f' % clf.score(X_train, y_train))
        print('f1-score: %.2f' % f1_score(y_train, y_hat, average='macro'))
        # joblib.dump(clf, '../model/20170410/user_0_{}.mod'.format(i))

        # clf = joblib.load('../model/20170410/user_0_{}.mod'.format(i))
        print('test >>>')
        print('预测结果:', clf.predict(X_test))
        print('实际结果:', y_test)
        print('准确率:', clf.score(X_test, y_test))
        print()
