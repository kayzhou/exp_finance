# -*- coding: utf-8 -*-
__author__ = 'Kay'

import pandas as pd
from sklearn.datasets import dump_svmlight_file

def csv2svmlib(in_name, out_name):
    data = pd.read_csv(in_name, header=None)
    y = data[0]
    X = data.ix[:, 1:]
    dump_svmlight_file(X, y, out_name)


csv2svmlib('lag10.csv', 'output.dat')