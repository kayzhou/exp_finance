__author__ = 'Kay Zhou'

import pandas as pd
import datetime
import matplotlib.pyplot as plt


def get_ratio_joy_fear(in_file, out_file):
    '''
    RJF -> ratio of joy to fear
    :param in_file:
    :return:
    '''
    data = pd.read_csv(in_file, index_col=None)
    # 选择时间段，论文中使用至2015-12-07
    data = data[data.date <= '2015-12-07']
    ratio_joy_fear = data.joy / data.fear
    print(ratio_joy_fear)
    data['RJF'] = ratio_joy_fear
    data.to_csv(out_file)


if __name__ == '__main__':
    # get_ratio_joy_fear('data/emotion_foll_level_0_pro.csv', 'data/exp_foll_level_0.csv')
    # get_ratio_joy_fear('data/emotion_foll_level_1_pro.csv', 'data/exp_foll_level_1.csv')
    # get_ratio_joy_fear('data/emotion_foll_level_2_pro.csv', 'data/exp_foll_level_2.csv')

    get_ratio_joy_fear('data/emotion_gender_f_pro.csv', 'data/exp_gender_f.csv')
    get_ratio_joy_fear('data/emotion_gender_m_pro.csv', 'data/exp_gender_m.csv')
