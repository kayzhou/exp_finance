# -*- coding: utf-8 -*-
__author__ = 'Kay'

import numpy as np
import pandas as pd
from dateutil.parser import parse

data = pd.read_csv('../data/20141201-20160501_hour_mood.csv')
print(data.shape)

# 获取某一时段的数据
d = data[data['date'] >= "2015-05-01"][data['date'] < "2015-09-01"]
print(d['fear00'])
