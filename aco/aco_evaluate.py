import numpy as np
import matplotlib.pyplot as plt
import pickle

which_problem = 3
ants = 150
# define Baselines
baselines = {'mzn' : [3980, 3258, 2998], 'aco' : [3632, 2878, 2617]}
mzn_baseline = baselines['mzn'][which_problem - 1]
aco_baseline = baselines['aco'][which_problem - 1]

results = "results_tsp_"+str(which_problem)+"_ants_"+str(ants)+".pickle"
data = pickle.load(open(results, "rb"))

for i in range(5):
    fig, axes = plt.subplots(1, 3, sharey=True, sharex=True, figsize = (15,10))

    overall_min = 2500
    overall_max = 5000
    c = 0
    t=0
    k=3
    lines = []
    names = []
    for j in range(i,44+i,5):
        if c-k == 0:
            k= k+3
            t = t+1
        stats, description = data[j]
        stats_mat = np.asarray(stats)
        if np.min(stats_mat[:,2]) < overall_min:
            overall_min = np.min(stats_mat[:,2])
        if np.max(stats_mat[:,2]) > overall_max:
            overall_max = np.max(stats_mat[:,2])
        mean_history = stats_mat[0]
        #max_history = stats_mat[1]
        min_history = stats_mat[:,2]
        minimum, = axes[t].plot(min_history, label= 'Q = '+str(description[3]))


        c = c+1
        lb = axes[t].axhline(y=mzn_baseline, label = 'minizinc', color = 'k')
        ln = axes[t].axhline(y=aco_baseline, label='aco', color = 'r')

    plt.ylim(overall_min, overall_max+500)
    axes.flatten()[-2].legend(loc='upper left', bbox_to_anchor=(-0.02, 1), ncol = 3)
    stats, description = data[i]

    plt.locator_params(nbins=4)
    plt.suptitle('alpha '+ str(description[0])+', beta '+ str(description[1])+ ', evaporation rate 0.3,0.5,0.7' ,size=16)
    fig.show()
    fig.savefig('results_problem_'+str(which_problem)+'_'+str(i)+'.png')

# plot only for fixed evaporation rate und intensity
c = 0
t=0
fig, axes = plt.subplots(2, 2, sharey=True, sharex=True, figsize = (15,10))
overall_min = 2500
overall_max = 10000

for i in range(1,5):

    print(t,c)
    stats, description = data[i]
    print(description)
    stats_mat = np.asarray(stats)
    if np.min(stats_mat[:,2]) < overall_min:
        overall_min = np.min(stats_mat[:,2])
    if np.max(stats_mat[:,1]) > overall_max:
        overall_max = np.max(stats_mat[:,1])
    mean_history = stats_mat[:,0]
    max_history = stats_mat[:,1]
    min_history = stats_mat[:,2]
    minimum, = axes[t][c].plot(min_history, label= 'best solution per iteration')
    #maximum, = axes[t][c].plot(mean_history, label= 'maximum solution per iteration')
    mean, = axes[t][c].plot(mean_history, label= 'mean per iteration')

    lb = axes[t][c].axhline(y=mzn_baseline, label = 'minizinc', color = 'k')
    ln = axes[t][c].axhline(y=aco_baseline, label='aco-reference', color = 'r')
    if c == 1:
        c=0
        t = 1
    else:
        print('here')
        c = c+1
        print(c)

plt.ylim(overall_min, 10000)
axes.flatten()[-2].legend(loc='upper left', bbox_to_anchor=(-0.02, 1), ncol = 3)
stats, description = data[i]

#plt.locator_params(nbins=4)
plt.suptitle('alpha '+ str(description[0])+', beta 1,5,10,15'+ ', evaporation rate 0.3, Q 1' ,size=16)
fig.show()
fig.savefig('results_problem_eva_fixed'+str(which_problem)+'.png')

# first plot
fig = plt.figure()
stats, description = data[0]
stats_mat = np.asarray(stats)
mean_history = stats_mat[:,0]
max_history = stats_mat[:,1]
min_history = stats_mat[:,2]
plt.plot(min_history, label= 'best solution per iteration')
plt.plot(mean_history, label= 'mean per iteration')

plt.axhline(y=mzn_baseline, label = 'minizinc', color = 'k')
plt.axhline(y=aco_baseline, label='aco-reference', color = 'r')

plt.ylim(2500, 30000)
axes.flatten()[-2].legend(loc='upper left', bbox_to_anchor=(-0.02, 1), ncol = 3)
stats, description = data[i]

#plt.locator_params(nbins=4)
plt.suptitle('alpha '+ str(1)+', beta '+ str(0)+ ', evaporation rate 0.3, Q 1' ,size=16)
fig.show()
fig.savefig('results_problem_eva_fixed_beta_1'+str(which_problem)+'.png')
