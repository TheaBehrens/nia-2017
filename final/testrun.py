import numpy as np
import time
import capacitated_aco_several_pheromones as aco


# some parameters...
INITIAL_PHEROMONE = 0.5

# During each iteration as many tours as specified in BATCH_SIZE are found
# Pheromone matrices updated only after the iteration for all
ITERATIONS = 30

# after how many runs the pheromone-trails are updated
# same size as there are customers seems reasonable
BATCH_SIZE = 100 

# how many good vehicle assingments of current batch
# to reuse in the next batch
# --> good to have a fraction of BATCH_SIZE
# this way some good solutions are kept, but it is still possible to explore new combinations
# it seems more reasonable that good vehicle combinations can be successfully applied again, if the choice of the first node is free and not randomized through ENFORCE_DIVERSE_START
KEEP_VEHICLES = int(np.floor(0.2 * BATCH_SIZE))

# can specify what fraction of iterations enforces some diversity among the found solutions
# later during training it is probably more reasonable to allow free choice, to take the ones that are actually good
# but it might make sense to enforce diversity in the beginning, to better explore the possibilities
# if DIVERSE_VEHICLES is True, it enforces to consider all types of vehicles
# else it enforces all customers to be considered a starting point during the batch
ENFORCE_DIVERSE_START = 0.7

# to force the algorithm to consider all types of vehicles for the first fraction of iterations
DIVERSE_VEHICLES = True


tic = time.clock()

aco.initialize(problem_path='problem1/', initial_pheromone=INITIAL_PHEROMONE)
aco.do_iterations(iterations=ITERATIONS, 
                  batch_size=BATCH_SIZE, 
                  keep_v=KEEP_VEHICLES,
                  enforce_diverse_start=ENFORCE_DIVERSE_START,
                  all_vehicles=DIVERSE_VEHICLES)

toc = time.clock()

print('it took so many seconds to compute: ', toc - tic)

