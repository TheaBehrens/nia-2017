import numpy as np
import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# data is a list of (stats, info) tuples
# where info is a string and stats is a fitness matrix with dims 
# TRIALxGENERATIONx2 (2=max, mean --> fitness of the population on that step)

data  = pickle.load(open("results_makespan2_short.pickle", "rb"))

fig, axes = plt.subplots(4, 2, sharey=True, sharex=True)
overall_min = 0
overall_max = -10000
for i in range(2):
    for j in range(4):
        stats, info = data[i*4+j]
        stats_mat = np.asarray(stats)
        if np.min(stats_mat) < overall_min:
            overall_min = np.min(stats_mat)
        if np.max(stats_mat) > overall_max:
            overall_max = np.max(stats_mat)
        mean_history = np.mean(stats_mat[:,:,1], axis=0)
        best_history = np.mean(stats_mat[:,:,0], axis=0)
        axes[j][i].plot(mean_history)
        axes[j][i].plot(best_history)
        axes[j][i].set_title(str(i*4+j))
       # print(info)

plt.ylim(overall_min, overall_max+100)
plt.locator_params(nbins=4)
plt.suptitle('Selection: tournament (size=10, n=35)', size=16)
fig.show()
fig.savefig('results2_short1.pdf')

# --------------------------------------------------------------
# same again, for second half of data (it's too crowded if put in one figure)
fig, axes = plt.subplots(4, 2, sharey=True, sharex=True)
overall_min = 0
overall_max = -10000
for i in range(2):
    for j in range(4):
        stats, info = data[i*4+j+8]
        stats_mat = np.asarray(stats)
        if np.min(stats_mat) < overall_min:
            overall_min = np.min(stats_mat)
        if np.max(stats_mat) > overall_max:
            overall_max = np.max(stats_mat)
        mean_history = np.mean(stats_mat[:,:,1], axis=0)
        best_history = np.mean(stats_mat[:,:,0], axis=0)
        axes[j][i].plot(mean_history)
        axes[j][i].plot(best_history)
        axes[j][i].set_title(str(i*4+j+8))
        #print(info)
plt.ylim(overall_min, overall_max+100)
plt.locator_params(nbins=4)
plt.suptitle('Selection: Roulette (n=35)', size=16)
fig.show()
fig.savefig('results2_short2.pdf')

