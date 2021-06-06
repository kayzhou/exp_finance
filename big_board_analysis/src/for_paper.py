# -*- coding: utf-8 -*-
__author__ = 'Kay'


def mark2emotion(mark):
    if mark == 1: return 'anger'
    elif mark == 2: return 'disgust'
    elif mark == 3: return 'happy'
    elif mark == 4: return 'sadness'
    elif mark == 5: return 'fear'


def for_table_corr(in_name):
    in_file = open(in_name)
    c = 1
    for line in in_file:
        line = line.strip().split(' ')
        result = str(c) + ' & '
        for i in range(len(line)):
            if i == len(line) - 1:
                result += '(%.2f)' % float(line[i]) + ' \\\\'
            else:
                result += '(%.2f)' % float(line[i]) + ' & '
        print(result); c += 1

# for_table_corr('/Users/Kay/Project/ML/big_board_analysis/data/proportion/pearson/4.txt')
for_table_corr('../data/proportion/shuffle_pearson/3.txt')

