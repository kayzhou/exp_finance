__author__ = 'Kay Zhou'

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# train_data = np.loadtxt('../data/proportion/训练数据/20141201-20150916_比例_三分类_true.txt')
# test_data = np.loadtxt('../data/proportion/测试数据/20150917-20151207_比例_三分类_true.txt')
# train_data = np.loadtxt('../data/proportion/训练数据/user_0_二分类.txt')
# test_data = np.loadtxt('../data/proportion/测试数据/user_0_二分类.txt')
train_data = np.loadtxt('../data/proportion/训练数据/user_0.csv')
test_data = np.loadtxt('../data/proportion/测试数据/user_0.csv')
X_train = train_data[:, 5:]
X_test = test_data[:, 5:]

for i in range(5):
    # clf = SVC(C=4, gamma=1)
    clf = RandomForestClassifier()
    # clf = GaussianNB()
    y_train = train_data[:, i]
    y_test = test_data[:, i]
    clf.fit(X_train, y_train)
    # print('预测结果：', clf.predict(X_test))
    # print('真实数据：', y_test)
    print(clf.score(X_test, y_test))
    # print(clf.score(X_test, y_test))
