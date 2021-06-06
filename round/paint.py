# ---------------------------------------------------
# File Name: paint.py
# Author: Kay Zhou
# mail: kayzhou.mail@gmail.com
# Created Time: 2017-09-11 21:08
# ---------------------------------------------------
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set(style="dark", palette="Set1")

data = {}
files = ['201610', '201611', '201612', '201701', '201702']
for f in files:
    d = json.load(open('data/{}_count_result.txt'.format(f)))
    data.update(d)
data = sorted(data.items(), key=lambda d: d[0])


def time_series():
    sh = pd.read_csv('data/SH#999999.txt')
    sh = sh[sh.date >= '2016-09-30'][sh.date < '2017-03-01']
    sh['return'] = (sh['close'] / sh['close'].shift(1) - 1) * 100
    sh = sh.dropna(axis=0)
    print(sh)
    # 时间序列
    x = [datetime.datetime.strptime(k[0], '%Y%m%d') for k in data]
    # print(len(ts_data), len(y))
    # ts_data['round'] = y
    # print(ts_data.corr())
    for i in range(0, 10):
        y = [v[1][0][i] / sum(v[1][0]) for v in data]
        if i == 0:
            sh['round'] = y
        plt.plot(x, y, label=str(i))

    print(sh)
    print(sh.corr())
    plt.legend(loc='best')
    plt.ylim(0, 0.2)
    plt.grid(True)
    # plt.show()


def bar():
    # 总体
    x = [datetime.datetime.strptime(k[0], '%Y%m%d') for k in data]
    count = [0] * 10
    for i in range(10):
        for d in data:
            count[i] = count[i] + d[1][0][i]
    print(count)

    count = [c / sum(count) for c in count]
    ax = plt.bar(range(10), count)
    plt.xticks(range(0, 10), [str(i) for i in range(0, 10)])
    plt.grid(True)
    plt.show()


def main():
    time_series()


if __name__ == '__main__':
    main()



