# -*- coding: utf-8 -*-
__author__ = 'Kay'

from sklearn.svm import SVR
from sklearn import datasets, linear_model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dts
import datetime


data = np.loadtxt('/Users/Kay/Project/ML/big_board_analysis/data/2016/20141208-20160309_day_lag_5.dat')
# train_X = data[:200, 6:]
# train_y = data[:200, 5]
# test_X = data[200:250, 6:]
# test_y = data[200:250, 5]

# train_X = data[:250, 6:]
# train_y = data[:250, 5]
# test_X = data[250:, 6:]
# test_y = data[250:, 5]

train_X = data[:200, 6:]
train_y = data[:200, 1]
test_X = data[200:, 6:]
test_y = data[200:, 1]


clf = linear_model.LinearRegression()
clf.fit(train_X, train_y)
print(clf.score(train_X, train_y))
print(clf.score(test_X, test_y))
predict_y = clf.predict(test_X)




# plt.plot(test_y)
# plt.plot(predict_y)
plt.plot(np.hstack((train_y[-50:], test_y)))
plt.plot(np.hstack((train_y[-50:], predict_y)))
# plt.plot(np.hstack((train_y, test_y)))
# plt.plot(np.hstack((train_y, predict_y)))
# plt.plot(test_y)
# plt.plot(predict_y)
# plt.ylim(-10, 10)
plt.show()

# def strs2dates(strs):
#     return [datetime.datetime.strptime(str(s)[:-2], '%Y%m%d') for s in strs]
#
# train_dates = strs2dates(train_data[train_items:, 0])
# test_dates = strs2dates(test_data[:, 0])
#
#
# figure = plt.figure()
# ax = figure.add_subplot(111)
# # ax.plot_date(np.hstack((train_dates, test_dates)), np.hstack((train_y, test_y)), 'b-', linewidth=1, alpha=0.8, )
# # ax.plot_date(test_dates, test_y, 'b-', linewidth=1, alpha=0.8)
# ax.plot_date(test_dates, predict_y, 'g-', linewidth=1, alpha=0.8)
# ax.grid()
# ax.xaxis.set_major_locator(dts.MonthLocator())
# ax.xaxis.set_major_formatter(dts.DateFormatter('%m\n%d'))
# # figure.suptitle('%s ~ %s' % (dates[0], dates[-1]), fontdict={'size': 16})
# # plt.ylim(-10, 10)
# plt.ylabel('Amount')
# plt.xlabel('Date')
# figure.autofmt_xdate()
# plt.figure()