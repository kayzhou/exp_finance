# -*- coding: utf-8 -*-
__author__ = 'Kay Zhou'

import os
import json

'''
{"sell": 0, "gender": "m", "buy": 0, "verified": "False", "uid": "1686384524",
 "retweedted": 0, "angry": 0, "rise": 0, "original": 1, "disgusted": 0,
 "status_count": "1131", "time": "Wed Dec 10 09:58:06 +0800 2014",
 "follower_count": "226", "fall": 0, "sad": 0, "scared": 0, "happy": 1}
'''

def get_gender(in_dir):
    d = {}
    gay = open('data/gay.txt', 'w')
    for in_file in os.listdir(in_dir):
        in_file = os.path.join(in_dir, in_file)
        for i, line in enumerate(open(in_file)):
            if i % 1000 == 0:
                print(i)
            line = json.loads(line.strip())
            uid = line['uid']
            gender = line['gender']
            if uid in d:
                if gender != d[uid]:
                    gay.write(uid + '\n')
                    d[uid] = gender
                    # print('Gay!')
            else:
                d[uid] = gender

    gay.close()
    json.dump(d, open('uid_gender.json', 'w'), indent=4)


if __name__ == '__main__':
    get_gender('data/result')
