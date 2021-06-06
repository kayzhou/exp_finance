# -*- coding: utf-8 -*-
__author__ = 'Kay Zhou'

import os
import json


def get_verify(in_dir):
    d = {}
    for in_file in os.listdir(in_dir):
        in_file = os.path.join(in_dir, in_file)
        for i, line in enumerate(open(in_file)):
            if i % 1000 == 0:
                print(i)
            line = json.loads(line.strip())
            uid = line['uid']
            v = 0 if line['verified'] == 'False' else 1
            if uid in d:
                if v != d[uid]:
                    d[uid] = v
                    # print('Gay!')
            else:
                d[uid] = v

    json.dump(d, open('uid_gender.json', 'w'), indent=4)


if __name__ == '__main__':
    get_verify('data/result')
