'''
性别区分
粉丝数区分 log10
'''

import os
import math
import json
import pytz
from dateutil.parser import parse
import collections
tz_cn = pytz.timezone("Asia/Shanghai")

# 全局结果
results = collections.OrderedDict()


def init_data():
    res = {
        'm':    [0] * 6,
        'f':    [0] * 6,
        'fol0': [0] * 6,
        'fol1': [0] * 6,
        'fol2': [0] * 6,
        'fol3': [0] * 6,
        'fol4': [0] * 6,
        'fol5': [0] * 6,
        'fol6': [0] * 6,
        'fol7': [0] * 6,
        'fol8': [0] * 6,
        'fol9': [0] * 6,
    }
    return res


def get_gender(d):
    return d['w:gender']


def get_follow_class(d):
    cnt = float(d['w:followers_count'])
    # print('cnt =', cnt)
    if cnt == 0:
        return 0
    elif cnt > 10 ** 9:
        return 9
    else:
        return int(math.log10(cnt))


def get_mood(d):
    return int(d['w:mood'])


def file_to_features(in_name):
    count = 0
    for line in open(in_name):
        count += 1
        d = json.loads(line.strip())
        gen = get_gender(d)
        fol = 'fol' + str(get_follow_class(d))
        mood = get_mood(d)
        dt = parse(d['w:created_at']).strftime('%Y-%m-%d')
        if dt not in results:
            results[dt] = init_data()

        if mood != -1:
            results[dt][gen][mood] += 1
            results[dt][fol][mood] += 1
        results[dt][gen][5] += 1
        results[dt][fol][5] += 1
        if not count % 10000:
            print(count)


if __name__ == '__main__':
    in_dir = '/home/kayzhou/exp/come_on_data/get_weibo/data/dapan'
    for in_name in os.listdir(in_dir):
        # if in_name != '成分指数.txt':
        #   continue
        in_name = os.path.join(in_dir, in_name)
        print(in_name)
        file_to_features(in_name)

    with open('results_user_class.txt', 'w') as f:
        for k, v in results.items():
            f.write(k + ' @@ ' + str(v) + '\n')


