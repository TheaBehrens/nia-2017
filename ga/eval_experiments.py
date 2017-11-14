import numpy as np
import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# data is a list of (stats, desc) tuples
# where info is a string and stats is a fitness matrix with dims TRIALxGENERATIOn
data  = pickle.load(open("results.pickle", "rb"))
fig, axes = plt.subplots(4, 4, sharey=True)

for i in range(4):
	for j in range(4):
		stats, info = data[i*4+j]
		mean_history = np.mean(stats, axis=0)
		axes[i][j].plot(mean_history)

fig.show()