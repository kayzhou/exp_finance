__author__ = 'Kay Zhou'

import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np


def compute_loss_aversion():
    '''
    计算损失厌恶指数

    re_emotion / return

    :return:
    '''

    d = pd.read_csv('data/VIP_day.csv')
    # print(d.close)

    # 一般收益率
    log_return = d.close

    # 对数收益率
    # print(len(d))
    # log_return = np.log(d['index'])
    # log_return = np.diff(log_return, 1)
    # print(log_return)
    # print(len(log_return))

    # 相对情绪
    # rela_anger = ((1 + d.joy) / (1 + d.anger)).apply(math.log)[1:]
    # rela_disgust = ((1 + d.joy) / (1 + d.disgust)).apply(math.log)[1:]
    # rela_sadness = ((1 + d.joy) / (1 + d.sadness)).apply(math.log)[1:]
    # rela_fear = ((1 + d.joy) / (1 + d.fear)).apply(math.log)[1:]
    rela_anger = ((1 + d.joy) / (1 + d.anger)).apply(math.log)
    rela_disgust = ((1 + d.joy) / (1 + d.disgust)).apply(math.log)
    rela_sadness = ((1 + d.joy) / (1 + d.sadness)).apply(math.log)
    rela_fear = ((1 + d.joy) / (1 + d.fear)).apply(math.log)

    # 损失厌恶指数
    loss_anger = rela_anger / log_return
    loss_disgust = rela_disgust / log_return
    loss_sadness = rela_sadness / log_return
    loss_fear = rela_fear / log_return

    # 组织数据
    data = pd.DataFrame({'date': d.date, 'loss_anger': loss_anger,
                         'loss_disgust': loss_disgust, 'loss_sadness': loss_sadness,
                         'loss_fear': loss_fear})

    data.to_csv('data/loss_aversion.csv')


def compute_state():
    '''
    计算损失厌恶指数

    emotion = state + reaction + noise

    reaction = f(return)

    return >= 0
    参数: -0.0172403725089 0.0290586156508

    return < 0
    参数: 0.0644436243832 -0.026442296003

    :return:
    '''
    def get_reaction(re):
        '''
        获取即时反应
        :param re:
        :return:
        '''
        if re >= 0:
            reaction = re * -0.0172
        elif re < 0:
            reaction = re * 0.0644
        # if re >= 0:
        #     reaction = re * -0.0172 + 0.0291
        # elif re < 0:
        #     reaction = re * 0.0644 - 0.0264
        # if re >= 0:
        #     reaction = re * 0.04
        # elif re < 0:
        #     reaction = re * -0.06
        else:
            reaction = None
        # reaction = 0
        return reaction


    d = pd.read_csv('data/VIP_day.csv')
    # print(d.close)

    # 收益率
    reaction = (d.close).apply(get_reaction)
    d_close = d.close

    # 相对情绪
    rela_anger = ((1 + d.joy) / (1 + d.anger)).apply(math.log)
    rela_disgust = ((1 + d.joy) / (1 + d.disgust)).apply(math.log)
    rela_sadness = ((1 + d.joy) / (1 + d.sadness)).apply(math.log)
    rela_fear = ((1 + d.joy) / (1 + d.fear)).apply(math.log)

    # 损失厌恶指数
    loss_anger = rela_anger / d_close
    loss_disgust = rela_disgust / d_close
    loss_sadness = rela_sadness / d_close
    loss_fear = rela_fear / d_close

    # 心理状态
    state = rela_fear - reaction

    # 组织数据
    data = pd.DataFrame({'date': d.date, 'loss_anger': loss_anger,
                         'loss_disgust': loss_disgust, 'loss_sadness': loss_sadness,
                         'loss_fear': loss_fear, 'state': state})

    data.to_csv('data/loss_aversion.csv')


def plot_loss():
    '''
    画损失厌恶指数和上证指数
    :return:
    '''
    d = pd.read_csv('data/VIP_day.csv')
    data = pd.read_csv('data/loss_aversion.csv')


    fig = plt.figure(figsize=(10, 6))

    # 损失厌恶
    ax1 = fig.add_subplot(111)
    # roll_state = data['state'].rolling(window=20).mean()
    # ax1.plot_date(d.date, roll_state, 'r', linewidth=1, label='loss aversion')

    ax1.plot_date(d.date, data.loss_fear, 'r', label='loss aversion')
    # ax1.plot_date(d.date, d.fear / 30000, 'g', linewidth=1, label='fear')

    # 上证指数
    ax2 = ax1.twinx()
    ax2.plot_date(d.date, d['index'], '-', linewidth=0.8, label='SH index')

    ax1.legend(loc='upper left')
    ax2.legend()
    # plt.show()
    plt.grid()
    plt.savefig('figure/loss_fear.pdf')

    # 保存数据
    _d = pd.DataFrame({'date': d.date, 'sh': d['index'], 'loss_fear': data.loss_fear, 'close': d['close']})
    _d.to_csv('data/loss_fear.csv')


if __name__ == '__main__':
    compute_loss_aversion()
    plot_loss()



