# -*- coding: utf-8 -*-
__author__ = 'Kay Zhou'

import os
import json
import pandas
import matplotlib.pyplot as plt

from dateutil.parser import parse
import pytz

tz_cn = pytz.timezone("Asia/Shanghai")


def read_uid_follower_level(in_name='data/uid_follower_level.json'):
    return json.load(open(in_name))


def read_uid_gender(in_name='data/uid_gender.json'):
    return json.load(open(in_name))


def read_uid_verify(in_name='data/uid_verify.json'):
    return json.load(open(in_name))


def count_emotions(in_dir='data/result', out_name='data/emotion.csv'):
    def get_emotion(w):
        return {
            'anger': w['angry'],
            'disgust': w['disgusted'],
            'joy': w['happy'],
            'sadness': w['sad'],
            'fear': w['scared']
        }

    def union_emotion(w1, w2):
        return {
            'anger': w1['anger'] + w2['anger'],
            'disgust': w1['disgust'] + w2['disgust'],
            'joy': w1['joy'] + w2['joy'],
            'sadness': w1['sadness'] + w2['sadness'],
            'fear': w1['fear'] + w2['fear']
        }

    d = {}

    for in_file in os.listdir(in_dir):
        in_file = os.path.join(in_dir, in_file)
        for i, line in enumerate(open(in_file)):
            if i % 1000 == 0:
                print(i)
            weibo = json.loads(line.strip())
            dt = parse(weibo['time']).strftime('%Y-%m-%d')
            emotion = get_emotion(weibo)

            if dt in d:
                d[dt] = union_emotion(d[dt], emotion)
            else:
                d[dt] = emotion

    data = sorted(d.items(), key=lambda d: d[0])
    out_file = open(out_name, 'w')
    out_file.write('date,anger,disgust,joy,sadness,fear\n')
    for i, d in enumerate(data):
        out_file.write(','.join([d[0], str(d[1]['anger']), str(d[1]['disgust']),
                                 str(d[1]['joy']), str(d[1]['sadness']), str(d[1]['fear'])]) + '\n')



def count_emotions_based_follower(in_dir='data/result'):
    '''
    根据不同的粉丝数计算情绪的统计
    >= 10000
    <= 100 < 10000
    < 100
    :return:
    '''
    def get_emotion(w):
        return {
            'anger': w['angry'],
            'disgust': w['disgusted'],
            'joy': w['happy'],
            'sadness': w['sad'],
            'fear': w['scared']
        }

    def union_emotion(w1, w2):
        return {
            'anger': w1['anger'] + w2['anger'],
            'disgust': w1['disgust'] + w2['disgust'],
            'joy': w1['joy'] + w2['joy'],
            'sadness': w1['sadness'] + w2['sadness'],
            'fear': w1['fear'] + w2['fear']
        }

    uid_follower_level = read_uid_follower_level()
    d_0 = {}
    d_1 = {}
    d_2 = {}  # 分别表示不同粉丝数等级的用户的情绪

    for in_file in os.listdir(in_dir):
        in_file = os.path.join(in_dir, in_file)
        for i, line in enumerate(open(in_file)):
            if i % 1000 == 0:
                print(i)
            weibo = json.loads(line.strip())
            uid = weibo['uid']
            level = uid_follower_level[uid]
            dt = parse(weibo['time']).strftime('%Y-%m-%d')
            emotion = get_emotion(weibo)

            if level == 0:
                if dt in d_0:
                    d_0[dt] = union_emotion(d_0[dt], emotion)
                else:
                    d_0[dt] = emotion
            elif level == 1:
                if dt in d_1:
                    d_1[dt] = union_emotion(d_1[dt], emotion)
                else:
                    d_1[dt] = emotion
            elif level == 2:
                if dt in d_2:
                    d_2[dt] = union_emotion(d_2[dt], emotion)
                else:
                    d_2[dt] = emotion

    json.dump(d_0, open('emotion_follower_level_0.json', 'w'), indent=4)
    json.dump(d_1, open('emotion_follower_level_1.json', 'w'), indent=4)
    json.dump(d_2, open('emotion_follower_level_2.json', 'w'), indent=4)


def count_emotions_based_gender(in_dir='data/result'):
    '''
    根据不同的粉丝数计算情绪的统计
    >= 10000
    <= 100 < 10000
    < 100
    :return:
    '''

    def get_emotion(w):
        return {
            'anger': w['angry'],
            'disgust': w['disgusted'],
            'joy': w['happy'],
            'sadness': w['sad'],
            'fear': w['scared']
        }

    def union_emotion(w1, w2):
        return {
            'anger': w1['anger'] + w2['anger'],
            'disgust': w1['disgust'] + w2['disgust'],
            'joy': w1['joy'] + w2['joy'],
            'sadness': w1['sadness'] + w2['sadness'],
            'fear': w1['fear'] + w2['fear']
        }

    uid_gender = read_uid_gender()
    d_f = {}
    d_m = {}  # 表示不同性别

    for in_file in os.listdir(in_dir):
        in_file = os.path.join(in_dir, in_file)
        for i, line in enumerate(open(in_file)):
            if i % 1000 == 0:
                print(i)
            weibo = json.loads(line.strip())
            uid = weibo['uid']
            gender = uid_gender[uid]
            dt = parse(weibo['time']).strftime('%Y-%m-%d')
            emotion = get_emotion(weibo)

            if gender == 'f':
                if dt in d_f:
                    d_f[dt] = union_emotion(d_f[dt], emotion)
                else:
                    d_f[dt] = emotion
            elif gender == 'm':
                if dt in d_m:
                    d_m[dt] = union_emotion(d_m[dt], emotion)
                else:
                    d_m[dt] = emotion

    convert_json_to_csv(d_f, 'emotion_gender_f.csv')
    convert_json_to_csv(d_m, 'emotion_gender_m.csv')


def count_emotions_based_verify(in_dir='data/result'):
    '''
    根据是否认证的粉丝数计算情绪的统计
    '''

    def get_emotion(w):
        return {
            'anger': w['angry'],
            'disgust': w['disgusted'],
            'joy': w['happy'],
            'sadness': w['sad'],
            'fear': w['scared']
        }

    def union_emotion(w1, w2):
        return {
            'anger': w1['anger'] + w2['anger'],
            'disgust': w1['disgust'] + w2['disgust'],
            'joy': w1['joy'] + w2['joy'],
            'sadness': w1['sadness'] + w2['sadness'],
            'fear': w1['fear'] + w2['fear']
        }

    uid_gender = read_uid_verify()
    d_f = {}
    d_m = {}  # 表示不同性别

    for in_file in os.listdir(in_dir):
        in_file = os.path.join(in_dir, in_file)
        for i, line in enumerate(open(in_file)):
            if i % 1000 == 0:
                print(i)
            weibo = json.loads(line.strip())
            uid = weibo['uid']
            gender = uid_gender[uid]
            dt = parse(weibo['time']).strftime('%Y-%m-%d')
            emotion = get_emotion(weibo)

            if gender == 1:
                if dt in d_f:
                    d_f[dt] = union_emotion(d_f[dt], emotion)
                else:
                    d_f[dt] = emotion
            elif gender == 0:
                if dt in d_m:
                    d_m[dt] = union_emotion(d_m[dt], emotion)
                else:
                    d_m[dt] = emotion

    convert_json_to_csv(d_f, 'emotion_verify_0.csv')
    convert_json_to_csv(d_m, 'emotion_verify_1.csv')


def convert_json_to_csv(json_data, out_name):
    '''
    无序的json输出到csv
    :param json_data:
    :param out_name:
    :return:
    '''
    out_file = open(out_name, 'w')
    out_file.write('date,anger,disgust,joy,sadness,fear\n')
    data = sorted(json_data.items(), key=lambda d: d[0])
    for i, d in enumerate(data):
        out_file.write(','.join([d[0], str(d[1]['anger']), str(d[1]['disgust']),
                                 str(d[1]['joy']), str(d[1]['sadness']), str(d[1]['fear'])]) + '\n')


def convert_abs_to_pro(in_name, out_name):
    '''
    绝对csv转比例csv
    :param in_name:
    :param out_name:
    :return:
    '''

    def read_trade_day(in_name='data/real-20141201-20151216.txt'):
        trade_day = set()
        for line in open(in_name):
            trade_day.add(line.split('\t')[0])
        return trade_day

    trade_day = read_trade_day()
    out_file = open(out_name, 'w')
    out_file.write('date,anger,disgust,joy,sadness,fear\n')
    for i, line in enumerate(open(in_name)):
        if i == 0:
            continue
        line = line.strip().split(',')
        date = line[0]
        if date not in trade_day:
            continue
        emotion = [line[1], line[2], line[3], line[4], line[5]]
        emotion = [float(e) for e in emotion]
        emotion = ['%.3f' % (e / sum(emotion)) for e in emotion]
        out_file.write(date + ',' + ','.join(emotion) + '\n')

    out_file.close()


if __name__ == '__main__':
    # count_emotions()
    # count_emotions_based_follower()
    # count_emotions_based_gender('data/result')
    count_emotions_based_verify('data/result')

    # convert_abs_to_pro('data/emotion.csv', 'data/emotion_pro.csv')
    # convert_abs_to_pro('data/emotion_foll_level_1.csv', 'data/emotion_foll_level_1_pro.csv')
    # convert_abs_to_pro('data/emotion_foll_level_2.csv', 'data/emotion_foll_level_2_pro.csv')
    # convert_abs_to_pro('data/emotion_gender_f.csv', 'data/emotion_gender_f_pro.csv')
    # convert_abs_to_pro('data/emotion_gender_m.csv', 'data/emotion_gender_m_pro.csv')
    convert_abs_to_pro('data/emotion_verify_0.csv', 'data/emotion_verify_0_pro.csv')
    convert_abs_to_pro('data/emotion_verify_1.csv', 'data/emotion_verify_1_pro.csv')
    # convert_abs_to_pro('data/emotion_verify_0.csv', 'data/emotion_verify_0_pro.csv')
    # convert_abs_to_pro('data/emotion_verify_1.csv', 'data/emotion_verify_1_pro.csv')
