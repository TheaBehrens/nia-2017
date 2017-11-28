import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import aco_thea as aco
import itertools, functools, operator
from functools import partial
import sys
import multiprocessing as mp

def evaluate_parameters(params):

    tic = time.clock()

    aco.initialize(which_problem, INITIAL_PHEROMONE)
    cost_in_iteration = np.zeros((iterations, 3))
    for i in range(iterations):
        solutions = aco.solution_generation(ants, alpha=params[2], beta=params[3])
        cost_mat = aco.pheromone_changes(solutions, evaporation_rate=params[0], Q=params[1])
        cost_in_iteration[i,:] = [np.mean(cost_mat), np.max(cost_mat), np.min(cost_mat)]
    toc = time.clock()
    #description = "alpha "+ str(params[2]) + " beta "+ str(params[3]) + " evaporation rate " + str(params[0]) + " Q " + str(params[1])
    description = [params[2],params[3], params[0], params[1]]

    print('it took so many seconds to compute: ', toc - tic)
    #print('best solution at last iteration: ', cost_in_iteration[-1,2])
    return cost_in_iteration, description

# specify here, problem, ant number and intial pheromone
which_problem = 1
ants= 150
INITIAL_PHEROMONE = 0.1
iterations = 100

# parameter grid!! change whatever you think is right
evaporation_rate = [0.1, 0.4, 0.6]
Q= [1,20, 50]
alpha= [1, 5]
beta= [1, 5]

pool = mp.Pool(4)
param_values = [evaporation_rate, Q, alpha, beta]
parameters = list(itertools.product(*param_values))
print(len(parameters))
results = list(pool.map(evaluate_parameters, parameters))
pickle.dump(results, open("results_tsp_"+str(which_problem)+"_ants_"+str(ants)+".pickle", "wb"))
