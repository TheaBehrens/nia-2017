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
# it seems more reasonable that good vehicle combinations can be successfully appplied again, if the choice of the first node is free and not randomized through ENFORCE_DIVERSE_STARTNODES
KEEP_VEHICLES = int(np.floor(0.2 * BATCH_SIZE))

# can specify what fraction of iterations enforces all nodes to be considered as start nodes
# later during training it is probably more reasonable to allow free choice of start nodes, to take the ones that are actually good
# but it might make sense to enforce diversity in the beginning, to better explore the possibilities
ENFORCE_DIVERSE_STARTNODES = 0.5



tic = time.clock()

aco.initialize(problem_path='problem1/', initial_pheromone=INITIAL_PHEROMONE)

aco.do_iterations(iterations=ITERATIONS, 
                  batch_size=BATCH_SIZE, 
                  keep_v=KEEP_VEHICLES,
                  enforce_diverse_start=ENFORCE_DIVERSE_STARTNODES)

toc = time.clock()

print('it took so many seconds to compute: ', toc - tic)

