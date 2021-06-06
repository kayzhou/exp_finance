#!/usr/bin/env python
# encoding: utf-8

# 我要统计0.1、0.01上面的数字import collections
import json
import os


def time_2_signal(t):
    if t >= '093000000' and t < '094500000':
        return 1
    elif t >= '094500000' and t < '100000000':
        return 2
    elif t >= '100000000' and t < '101500000':
        return 3
    elif t >= '101500000' and t < '103000000':
        return 4
    elif t >= '103000000' and t < '104500000':
        return 5
    elif t >= '104500000' and t < '110000000':
        return 6
    elif t >= '110000000' and t < '111500000':
        return 7
    elif t >= '111500000' and t < '113000000':
        return 8
    elif t >= '130000000' and t < '131500000':
        return 9
    elif t >= '131500000' and t < '133000000':
        return 10
    elif t >= '133000000' and t < '134500000':
        return 11
    elif t >= '134500000' and t < '140000000':
        return 12
    elif t >= '140000000' and t < '141500000':
        return 13
    elif t >= '141500000' and t < '143000000':
        return 14
    elif t >= '143000000' and t < '144500000':
        return 15
    elif t >= '144500000' and t <= '150000000':
        return 16


def amt():
    data = {}
    for in_name in os.listdir('trans'):
        in_name = os.path.join('trans', in_name)

        for i, line in enumerate(open(in_name)):

            if not i % 100000:
                print(in_name, i)

            if not line.startswith('[Trans]'):
                continue

            words = line.strip().split(',')
            dt = words[2]
            time = words[3]
            cancel = words[6]
            BS = words[8]
            price = words[9]
            amt = float('%.2f' % (float(words[10]) * float(price)))
            # print(amt)

            if float(price) <= 0:
                continue

            if (time >= '093000000' and time <= '113000000') or (time >= '130000000' and time <= '150000000'):
                list_p = price.split('.')
                v1 = int(list_p[1][0])
                v2 = int(list_p[1][1])

                # 时间信号，15分钟
                time_signal = time_2_signal(time)

                # 日期不存在在data中
                if dt not in data:
                    tmp = {
                        'overall': [0] * 10,
                        'B': [0] * 10,
                        'S': [0] * 10,
                        'C': [0] * 10,
                    }
                    for i in range(5, 10001, 5):
                        tmp[str(i) + '*10^4'] = [0] * 10

                    for i in range(1, 11):
                        tmp[str(i) + '*10^3'] = [0] * 10

                    for i in range(1, 17):
                        tmp['time-signal']

                    data[dt] = tmp

                # 全局
                data[dt]['overall'][v2] = data[dt]['overall'][v2] + amt

                # 买卖标记
                if BS == 'B':
                    data[dt]['B'][v2] = data[dt]['B'][v2] + amt
                elif BS == 'S':
                    data[dt]['S'][v2] = data[dt]['S'][v2] + amt
                elif cancel != '':
                    print('Canceled trans.')
                    data[dt]['C'][v2] = data[dt]['C'][v2] + amt

                # 量上的分析
                if amt >= 10000:
                    how_w = int(amt / 50000)
                    data[dt][str((how_w + 1) * 5) + '*10^4'][v2] += amt
                elif amt >= 1000:
                    how_t = int(amt / 1000)
                    data[dt][str(how_t + 1) + '*10^3'][v2] += amt

    data = sorted(data.items(), key=lambda d: d[0])
    json.dump(data, open('20170918-amt-count-result.txt', 'w'), indent=4)


def time_analysis():
   data = {}
   for in_name in os.listdir('trans'):
        in_name = os.path.join('trans', in_name)

        for i, line in enumerate(open(in_name)):

            if not i % 100000:
                print(in_name, i)

            if not line.startswith('[Trans]'):
                continue

            words = line.strip().split(',')
            dt = words[2]
            time = words[3]
            cancel = words[6]
            BS = words[8]
            price = words[9]
            amt = float('%.2f' % (float(words[10]) * float(price)))
            # print(amt)

            if float(price) <= 0:
                continue

            if (time >= '093000000' and time <= '113000000') or (time >= '130000000' and time <= '150000000'):
                list_p = price.split('.')
                v1 = int(list_p[1][0])
                v2 = int(list_p[1][1])

                # 时间信号，15分钟
                time_signal = time_2_signal(time)

                # 日期不存在在data中
                dt_add_signal = dt + '-' + time_2_signal
                if dt_add_signal not in data:
                   tmp = {
                        'times': [0] * 10,
                        'amt': [0] * 10,
                   }
                   data[dt] = tmp

                # 全局
                data[dt]['times'][v2] = data[dt]['times'][v2] + amt
                data[dt]['amt'][v2] = data[dt]['amt'][v2] + amt

    data = sorted(data.items(), key=lambda d: d[0])
    json.dump(data, open('20180131-count-time-result.txt', 'w'), indent=4)


if __name__ == '__main__':
    time_analysis()