# -*- coding: utf-8 -*-
__author__ = 'Kay Zhou'

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

'''

{"sell": 0, "gender": "m", "buy": 0, "verified": "False", "uid": "1686384524",
 "retweedted": 0, "angry": 0, "rise": 0, "original": 1, "disgusted": 0,
 "status_count": "1131", "time": "Wed Dec 10 09:58:06 +0800 2014",
 "follower_count": "226", "fall": 0, "sad": 0, "scared": 0, "happy": 1}

'''


def get_followers_count(in_dir):
    d = {}
    for in_file in os.listdir(in_dir):
        in_file = os.path.join(in_dir, in_file)
        for i, line in enumerate(open(in_file)):
            if i % 1000 == 0:
                print(i)
            line = json.loads(line.strip())
            uid = line['uid']
            count = int(line['follower_count'])
            if uid in d:
                if count > d[uid]:
                    d[uid] = count
            else:
                d[uid] = count

    json.dump(d, open('uid_followerCount.json', 'w'), indent=4)


def hist_followers_count(in_name='data/uid_followerCount.json'):
    '''
    粉丝数分布
    :param in_name:
    :return:
    '''
    data = json.load(open(in_name))
    d = pd.Series([np.log10(v) for v in data.values() if v > 10])
    plt.xlabel('$log_{10}(follower)$', fontsize=18)
    plt.ylabel('$Frequency$', fontsize=20)
    n, bins, patches = plt.hist(x=d, range=(1, 7), bins=30, alpha=0.9, rwidth=0.8)
    plt.grid()
    # plt.cla()
    # print(n, bins)
    # plt.plot(bins[:-1], n, 'black')
    plt.show()
    # print(d.describe())


if __name__ == '__main__':
    # get_followers_count('data/result')

    hist_followers_count()
