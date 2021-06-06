#coding:utf-8

import os
import sys

di = '/home/kay/project/come_on_money/deep_learning_stock/stock-data'
for fi in os.listdir(di):
    in_name = di + '/' + fi
    if fi.startswith('6') or fi.startswith('0') or fi.startswith('3'):
        continue
    cmd = '/home/kay/anaconda3/bin/python stock_gru.py %s > ../out/%s.out' % (in_name, fi)
    print(cmd)
    os.system(cmd)
