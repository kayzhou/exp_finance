# -*- coding: utf-8 -*-
__author__ = 'Kay Zhou'

'''
{"sell": 0, "gender": "m", "buy": 0, "verified": "False", "uid": "1686384524",
 "retweedted": 0, "angry": 0, "rise": 0, "original": 1, "disgusted": 0,
 "status_count": "1131", "time": "Wed Dec 10 09:58:06 +0800 2014",
 "follower_count": "226", "fall": 0, "sad": 0, "scared": 0, "happy": 1}
'''

import os
import json
from dateutil.parser import parse
import pytz

tz_cn = pytz.timezone("Asia/Shanghai")


def get_retweet(in_dir, out_name):
    d = {}

    for in_file in os.listdir(in_dir):
        in_file = os.path.join(in_dir, in_file)
        for i, line in enumerate(open(in_file)):
            if i % 5000 == 0:
                print(i)
            weibo = json.loads(line.strip())
            dt = parse(weibo['time']).strftime('%Y-%m-%d')
            r = weibo['retweedted']

            if dt in d:
                d[dt][0] += r
                d[dt][1] += 1

            else:
                d[dt] = [r, 1]
    data = sorted(d.items(), key=lambda d: d[0])
    out_file = open(out_name, 'w')
    out_file.write('date,retweet,sum,rate\n')
    for i, d in enumerate(data):
        out_file.write(','.join([d[0], str(d[1][0]), str(d[1][1]), str(d[1][0] / d[1][1])]) + '\n')


if __name__ == '__main__':
    get_retweet('data/result', 'data/retweet.csv')
