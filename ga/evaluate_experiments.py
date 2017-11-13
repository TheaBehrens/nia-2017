#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 12:44:51 2017

@author: tbehrens, iibs
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('makespan_1.pickle','rb') as f:
    stats = pickle.load(f)

with open('makespan_1_times.pickle','rb') as f:
    times = pickle.load(f)
    
    
print(stats.shape)
print(times)

for i in range(8):
    plt.figure()
    plt.plot(stats[i,:,0])
    plt.hold(True)
    plt.plot(stats[i,:,1])
    plt.show()