#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 12:44:51 2017
@author: tbehrens, iibs
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('makespan_2_results.pickle','rb') as f:
    stats = pickle.load(f)

with open('makespan_2_times.pickle','rb') as f:
    times = pickle.load(f)
    
param_grid = [['random','random'],['roulette','n_best'],
              ['crossover','crossover'],['random','each_random'], 
              ['all','keep_best']]

limit_min = np.min(stats) - 30             
limit_max = np.max(stats) + 100

cnt = 0
for i in range(1):
    for j in range(2): #2   
        for k in range(1):
            for l in range(2): #2
                for m in range(2): #2
                    plt.figure()
                    plt.plot(stats[:,0,cnt], label='mean')
                    plt.hold(True)
                    #plt.plot(stats[:,1,cnt], linestyle='none', marker='.', label='best')
                    plt.plot(stats[:,1,cnt], label='best')
                    title_str = 'init: ' + param_grid[0][i] + ', '
                    title_str += 'select: '  + param_grid[1][j] + ', '
                    title_str += 'rec: '  + param_grid[2][k] + ',\n '
                    title_str += 'mut: '  + param_grid[3][l] + ', '
                    title_str += 'repl: '  + param_grid[4][m]
                    plt.title(title_str)
                    plt.ylim(limit_min, limit_max)
                    plt.legend()
                    save_name = str(cnt) + '.pdf'
                    plt.savefig(save_name)
                    cnt += 1
                    plt.show()





'''
for i in range(8): # 8
    plt.figure()
    plt.plot(stats[:,0,i], label='mean')
    plt.hold(True)
    plt.plot(stats[:,1,i], linestyle='none', marker='.', label='best')
    plt.legend()
    plt.show()
    #print('begin mean: ', stats[0,0,i], ' best: ', stats[0,1,i])
    #print('end mean: ', stats[-5:-1,0,i])
    #print(' best: ', stats[-5:-1,1,i])
    #print('------')
    '''
