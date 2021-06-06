#coding: utf-8
__author__ = 'Kay'

from dateutil.parser import *
import datetime, time


def make_timestamp(str_time):
    print("str_time", str_time)
    if str_time is None:
        print("make_timestamp: None")
        return '0'
    try:
        datetime = parse(str_time)
        timestamp = str(time.mktime(datetime.timetuple()))
        return timestamp
    except:
        print("make_timestamp: error format.")
        print(str_time)
        return '0'


def get_now_last_hour(rule):
    now = get_now(rule)
    dt = datetime.datetime.strptime(now, rule)
    time_delta = datetime.timedelta(hours=1)
    dt1 = dt - time_delta
    dt2 = dt1 - time_delta
    return dt1.strftime(rule), dt2.strftime(rule)


def get_now_last_day(rule):
    now = get_now(rule)
    dt = datetime.datetime.strptime(now, rule)
    time_delta = datetime.timedelta(days=1)
    dt1 = dt - time_delta
    return dt1.strftime(rule)


def get_last_2_hour(dt, rule):
    dt1 = datetime.datetime.strptime(dt, rule)
    time_delta = datetime.timedelta(hours=2)
    dt2 = dt1 - time_delta
    return dt2.strftime(rule)     


def get_now(rule='%Y%m%d'):
    return time.strftime(rule, time.localtime(time.time()))


def add_day(str_datetime, n, rule='%Y%m%d'):
    '''
    :param str_datetime: 时间字符串
    :return: +n day 时间字符串
    '''
    dt = datetime.datetime.strptime(str_datetime, rule)
    time_delta = datetime.timedelta(days=n)
    dt = dt + time_delta
    return dt.strftime(rule)


def add_hour(str_datetime, n, rule='%Y%m%d%H'):
    '''
    :param str_datetime: 时间字符串
    :return: +n hour 时间字符串
    '''
    dt = datetime.datetime.strptime(str_datetime, rule)
    time_delta = datetime.timedelta(hours=1)
    dt = dt + time_delta
    return dt.strftime(rule)


def add_min(str_datetime, n, rule='%Y%m%d%H%M'):
    '''
    :param str_datetime: 时间字符串
    :return: +mins 时间字符串
    '''
    dt = datetime.datetime.strptime(str_datetime, rule)
    dt = dt + datetime.timedelta(minute=n)
    return dt.strftime(rule)


def get_date_list(s, e, rule='%Y%m%d'):
    start = datetime.datetime.strptime(s, rule)
    end = datetime.datetime.strptime(e, rule)
    dt = start
    date_list = []
    while dt < end:
        date_list.append(dt.strftime(rule))
        dt = add_day(dt, 1, rule)
    return date_list


if __name__ == '__main__':
    print(add_min('201509111145', -45, '%Y%m%d%H%M'))
