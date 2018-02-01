import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import capacitated_aco as aco


# some parameters...
INITIAL_PHEROMONE = 0.1
ants = 15
iterations = 13
evaporation_rate = 0.4
Q = 100
alpha=3
beta=5



tic = time.clock()

aco.initialize('problem1/', INITIAL_PHEROMONE)
aco.do_iterations(70)
# aco.collect_several_solutions()

toc = time.clock()

print('it took so many seconds to compute: ', toc - tic)

