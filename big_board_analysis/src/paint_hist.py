# -*- coding: utf-8 -*-
__author__ = 'Kay'

import numpy as np
import matplotlib.pyplot as plt
import kmeans

'''
画数据分布图以及聚类中心
'''

arr = np.loadtxt('../data/proportion/normalization.txt', usecols=(3,))
n, bins, patches = plt.hist(arr, 20, alpha=0.60)
center = kmeans.kmeans_boundary(arr, 3)

plt.plot(np.array([center[0][0]] * 100), np.arange(0, 20.0, 0.2), 'r--', linewidth=1)
plt.plot(np.array([center[1][0]] * 100), np.arange(0, 20.0, 0.2), 'r--', linewidth=1)
plt.plot(np.array([center[2][0]] * 100), np.arange(0, 20.0, 0.2), 'r--', linewidth=1)

plt.show()
