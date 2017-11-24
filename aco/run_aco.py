import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import aco_thea as aco


# some parameters...
INITIAL_PHEROMONE = 0.1
ants = 15
iterations = 100
evaporation_rate = 0.4
which_problem = 3
Q = 100
alpha=3
beta=5
savename = '24nov_' +str(ants)+ 'ants_' +str(iterations)+ 'iterations_' +str(evaporation_rate) +'roh_' +str(Q)+ 'q_' +str(alpha) + 'a_' +str(beta) + 'b' 


tic = time.clock()

aco.initialize(which_problem, INITIAL_PHEROMONE)
cost_in_iteration = np.zeros((iterations, 3))
for i in range(iterations):
    solutions = aco.solution_generation(ants, alpha=alpha, beta=beta)
    cost_mat = aco.pheromone_changes(solutions, evaporation_rate=evaporation_rate, Q=Q)
    cost_in_iteration[i,:] = [np.mean(cost_mat), np.max(cost_mat), np.min(cost_mat)]

toc = time.clock()

print('it took so many seconds to compute: ', toc - tic)
print('best solution at last iteration: ', cost_in_iteration[-1,2])

with open(savename+'.pickle', 'wb') as f:
    pickle.dump(cost_in_iteration, f, pickle.HIGHEST_PROTOCOL)

fig = plt.figure()
plt.plot(cost_in_iteration[:,0])
fig.show()
fig.savefig(savename + '.pdf')

