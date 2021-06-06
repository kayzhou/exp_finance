__author__ = 'Kay Zhou'

'''
统计每个关键词每天的词频情况

可以从MongoDB拿，

也可以从beihang17:/home/kayzhou/exp/come_on_data/get_weibo/data/dapan或stock20141201-20161017中拿；
'''

import os
import json
from collections import Counter
from dateutil.parser import parse
import pytz

tz_cn = pytz.timezone("Asia/Shanghai")


def get_word_cloud(in_dir, out_name):
    d = {}
    # 每日词频统计临时变量
    for in_file in os.listdir(in_dir):
        if not in_file.endswith('.txt'):
            continue
        print(in_file)
        in_file = os.path.join(in_dir, in_file)
        for i, line in enumerate(open(in_file)):
            if i % 1000 == 0:
                print(i)
            line = json.loads(line.strip())
            # 时间
            dt = parse(line['w:created_at']).strftime('%Y-%m-%d')
            # 分词结果
            seg = line['w:seg'].split(' ')
            # print(Counter(seg))

            # 已经存在
            if dt in d:
                d[dt] = d[dt] + Counter(seg)
            # 不存在
            else:
                d[dt] = Counter(seg)

    data = sorted(d.items(), key=lambda d: d[0])
    out_file = open(out_name, 'w')
    for dt, c in data:
        out_file.write('![datetime]' + dt + '\n')
        for word, frequency in c.most_common(100):
            out_file.write(word + ',' + str(frequency) + '\n')


if __name__ == '__main__':
    get_word_cloud(in_dir='/home/kayzhou/exp/come_on_data/get_weibo/data/dapan',
                   out_name='data/word_frequency.txt')
