# -*- coding: utf-8 -*-
__author__ = 'Kay'

import numpy as np
import datetime
import time_tool


def str_2_datetime(in_str):
    # print(in_int)
    return datetime.datetime.intptime(in_str, '%Y%m%d %H%M%S').intftime('%c')


def load_dates(in_name = '../data/date.txt'):
    dates = []
    for line in open(in_name):
        line = line.strip()
        dates.append(line)
    return dates


def load_sh_history():
    in_file = open('../data/000001.sh_kmin_20141001_20151225-full.txt')
    data = {}
    for line in in_file:
        line = line.strip()
        items = line.split(', ')
        data[str_2_datetime(items[0])] = {'open': items[1], 'high': items[2], 'low': items[3], 'close': items[4],
                                          'vol': items[5], 'amount': items[6], 'deals': items[7]}
    in_file.close()
    return data


def get_sh_hour(data, dates, time_nodes):
    results = {}
    for i, d in enumerate(dates):
        # print(d)
        if i == 0: continue
        last_day_close = data[str_2_datetime(dates[i-1] + ' 142700')]['close']
        day_result = []
        for j, t in enumerate(time_nodes):
            # print(t)
            if j == 0:
                day_result.append('1' if data[str_2_datetime(d + ' ' + t)]['close'] >= last_day_close else '0')
            else:
                day_result.append('1' if data[str_2_datetime(d + ' ' + t)]['close'] >= data[str_2_datetime(d + ' ' + time_nodes[j-1])]['close'] else '0')
        results[d] = day_result
    return results


def load_mood():
    mood = {}
    in_file = open('/Users/Kay/Project/EXP/big_board_fine_grained/data/15mins.txt')
    for line in in_file:
        line = line.strip().split(' ')
        summ = int(line[1]) + int(line[2]) + int(line[3]) + int(line[4]) + int(line[5])
        mood[line[0]] = {'0': line[1], '1': line[2], '2': line[3],
                         '3': line[4], '4': line[5], 'sum': str(summ)}
    return mood


def get_mood(feature_nodes):
    # sh_data = load_sh_history()
    mood = load_mood()
    dates = load_dates()

    for d in dates:
        for t in feature_nodes:
            line = d + ' '
            for index in t:
                try:
                    # print(mood[d + index])
                    m = mood[d + index]
                    line += (m['0'] + ' ' + m['1'] + ' ' + m['2'] + ' ' + m['3'] + ' ' + m['4'] + ' ' + m['sum'] + ' ')
                    # print(d + index + ' ' + m['0'] + ' ' + m['1'] + ' ' + m['2'] + ' ' + m['3'] + ' ' + m['4'] + ' ' + m['sum'])
                except:
                    line += '0 0 0 0 0 0 '
                    # print(d + index + ' 0 0 0 0 0 0')
            print(line)


def make_features():
    gg = []
    for goal in open('../data/hour_data.txt'):
        gg += goal.strip().split(' ')[1:]

    out_file = open('really_features.txt', 'w')
    i = 0
    for line in open('../data/features_goal.txt'):
        line = line.strip()
        line += (' ' + str(gg[i]))
        out_file.write(line + '\n')
        i += 1


def obtain_svm_file():
    in_file = open('../data/really_features.txt')
    file_train = open('../data/SVM_hour.txt', 'w')
    for line in in_file:
        line = line.strip().split(' ')
        x = line[1:-1]
        file_train.write(line[-1] + ' ' + ' '.join([str(i+1) + ':' + str(xi) for i, xi in enumerate(x)]) + '\n')



if __name__ == '__main__':
    time_nodes = ['093000', '103000', '112900', '130000', '140000', '142700']
    feature_nodes = [['072', '073', '080', '081'],
                     ['082', '083', '090', '091'],
                     ['092', '093', '100', '101'],
                     ['102', '103', '110', '111'],
                     ['130', '131', '132', '133'],
                     ['140', '141', '142', '143']]
    # get_mood(feature_nodes)

    # dates = load_dates()
    # data = load_sh_history()
    # # print(data)
    # hour_data = get_sh_hour(data, dates, time_nodes)
    # hour_data = sorted(hour_data.items(), key=lambda d:d[0])
    # for k, v in hour_data:
    #     print k, ' '.join(v)

    # make_features()
    obtain_svm_file()