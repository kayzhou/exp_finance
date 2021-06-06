__author__ = 'Kay Zhou'

'''
分析不同收益率下，大家讨论的内容（词）是否有特点

'''

import pandas as pd
from collections import defaultdict


def union_all(in_name):
    '''
    合并所有日期的word
    :param in_name:
    :return:
    '''
    filter = get_filter_words()
    dict_word_count = defaultdict(int)
    for line in open(in_name):
        if line.startswith('![datetime]'):
            print(line)
            continue
        try:
            word, count = line.strip().split(',')
        except ValueError:
            print(line)
            continue

        dict_word_count[word] += int(count)

    # 词典按词频排序
    di = sorted(dict_word_count.items(), key=lambda d: d[1], reverse=True)

    with open('data/word_frequency_all.csv', 'w') as f:
        f.write('word,count\n')
        for k, v in di:
            if k in filter:
                continue
            f.write(k + ',' + str(v) + '\n')


def single_word(in_name='data/word_frequency_all.csv'):
    data = pd.read_csv(in_name)
    # print(data)
    for word in data.word:
        try:
            if len(word) == 2:
                print(word)
        except:
            print('error ->', word)


def get_filter_words(in_name='data/filter_word.txt'):
    '''
    获取过滤词
    :param in_name:
    :return:
    '''
    words = set({})
    for line in open(in_name):
        word = line.strip()
        words.add(word)

    return words


def write_words_with_day(in_name='../stat_stock_tweet/data/word_frequency.txt'):
    '''
    转化到每日的词频
    :param in_name:
    :return:
    '''
    filter = get_filter_words()
    # 命中日期
    day = 'init'
    list_word_count = []

    for line in open(in_name):

        if line.startswith('![datetime]'):
            print(line)
            line_day = line.strip().split('![datetime]')[1]
            # 新的开始，先收尾
            if line_day != day:
                with open('data/word/{}.csv'.format(day), 'w') as f:
                    f.write('word,count\n')
                    for word, count in list_word_count:
                        f.write(word + ',' + count + '\n')
                list_word_count = []
                day = line_day
        else:
            try:
                word, count = line.strip().split(',')
                if word in filter:
                    continue
                list_word_count.append((word, count))
            except:
                print('error ->', line)


def get_word_with_days(days, out_name):
    '''
    聚合天级别的词频
    :param days:
    :return:
    '''
    d = defaultdict(int)
    in_dir = 'data/word'
    for day in days:
        in_name = in_dir + '/' + day + '.csv'
        data = pd.read_csv(in_name)
        for i, row in data.iterrows():
            # print(row[0])
            word = row['word']
            cnt = row['count']
            # print(type(cnt))

            d[word] += int(cnt)

    d = sorted(d.items(), key=lambda k: k[1], reverse=True)
    with open(out_name, 'w') as f:
        f.write('word,count\n')
        for k, v in d:
            f.write(str(k) + ',' + str(v) + '\n')


def get_rise_fall_days():
    '''
    获取涨或者跌的日期
    :return:
    '''
    d = pd.read_csv('data/VIP_day.csv')
    rise = []
    fall = []
    for i, row in d.iterrows():
        if float(row['close']) >= 0:
            rise.append(row['date'])
        else:
            fall.append(row['date'])

    return rise, fall


if __name__ == '__main__':
    # union_all('../stat_stock_tweet/data/word_frequency.txt')
    # single_word()
    # write_words_with_day()

    # rise_days, fall_days = get_rise_fall_days()
    # get_word_with_days(rise_days, 'data/rise_days_word.csv')
    # get_word_with_days(fall_days, 'data/fall_days_word.csv')
    pass