# -*- coding: utf-8 -*-
__author__ = 'Kay'

import random
import sys
import json
import datetime
import time
from dateutil.parser import parse
import pytz

stock_keyword = ['涨', '跌', '买', '卖']
tz_cn = pytz.timezone("Asia/Shanghai")


# 累加操作
def union_result(re1, re2):
    result = re1
    for i in range(len(re2)):
        result[i] += re2[i]
    return result


# 统计出现股票关键词的数量
def how_many_stock_keyword(segs):
    result = [0] * len(stock_keyword)
    for word in segs:
        for i, s_word in enumerate(stock_keyword):
            if s_word in word:
                result[i] += 1
    return result


def count_stock_fi(fi_name):

    stat = dict()

    for line in open('/home/kayzhou/exp/come_on_data/get_weibo/data/dapan/' + fi_name + '.txt'):
        weibo = json.loads(line.strip())
        # print(weibo)
        dt = parse(weibo['w:created_at']).strftime('%Y-%m-%d')
        cnt = how_many_stock_keyword(weibo['w:seg'])

        if dt in stat:
            stat[dt] = union_result(stat[dt], cnt)
        else:
            print(dt)
            stat[dt] = cnt

    w_fi = open('data/keyword_' + fi_name + '.csv', 'w')

    count = 0
    stat = sorted(stat.items(), key=lambda d:d[0])
    for k, v in stat:
        count += 1
        w_fi.write(k + ',' + ','.join([str(vi) for vi in v]) + '\n')
    print(fi_name, '统计天数：', count)


def count_sum_retweet_fi(fi_name):
    '''
    计算每天微博量和转发量
    :param fi_name:
    :return:
    '''
    stat = dict()

    print(fi_name)

    for line in open('../../come_on_data/get_weibo/data/dapan/' + fi_name + '.txt'):
        weibo = json.loads(line.strip())
        dt = parse(weibo['w:created_at']).strftime('%Y-%m-%d')

        if dt in stat:
            if "w:retweeted_id" in weibo:
                stat[dt][1] += 1
            stat[dt][0] += 1
        else:
            print(dt)
            if "w:retweeted_id" in weibo:
                stat[dt] = [1, 1]
            else:
                stat[dt] = [1, 0]


    out_file = open('data/' + fi_name + '.csv', 'w')

    count = 0
    stat = sorted(stat.items(), key=lambda d:d[0])
    for k, v in stat:
        count += 1
        out_file.write(k + ',' + ','.join([str(vi) for vi in v]) + '\n')
    print(fi_name, '统计天数：', count)


def union_different_keywords(fis, out_name):
    '''
    合并关键词结果
    '''
    stat = dict()
    w_fi = open(out_name, 'w')

    for fi_name in fis:
        print(fi_name)
        for line in open('data/keyword_' + fi_name + '.csv'):
            line = line.strip()
            re = line.split(',')
            dt = re[0]
            cnts = re[1:]
            cnts = [int(c) for c in cnts]
            print(cnts)
            if dt in stat:
                stat[dt] = union_result(stat[dt], cnts)
            else:
                stat[dt] = cnts

    count = 0
    stat = sorted(stat.items(), key=lambda d:d[0])
    for k, v in stat:
        count += 1
        w_fi.write(k + ',' + ','.join([str(vi) for vi in v]) + '\n')
    w_fi.close()


def load_count_stock_keyword():
    '''
    2015-07-22 载入数据
    '''
    stat = dict()
    for line in open('result.dat'):
        line = line.strip()
        dt, cnt = line.split('\t')
        cnts = cnt.split(' ')
        cnts = [int(c) for c in cnts]
        stat[dt] = cnts
    return stat


if __name__ == '__main__':

    fis = ['成分指数', '股票', '股市', '上证指数', '深成指', '证券']
    # fis = ['上证指数']
    # for fi in fis:
    #     count_stock_fi(fi)
    #    count_sum_retweet_fi(fi)

    union_different_keywords(fis, 'keyword_results.csv')

