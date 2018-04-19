import numpy as np
import time
import itertools
import capacitated_aco_opt_end as aco


# some parameters...
#0.00001
INITIAL_PHEROMONE = [0.0001,0.3, 0.5, 0.7]
KEEP_VEHICLES = False
# During each iteration as many tours as specified in BATCH_SIZE are found
# Pheromone matrices updated only after the iteration for all
ITERATIONS = [1000]

# after how many runs the pheromone-trails are updated
# same size as there are customers seems reasonable
BATCH_SIZE = [100]


# can specify what fraction of iterations enforces some diversity among the found solutions
# later during training it is probably more reasonable to allow free choice, to take the ones that are actually good
# but it might make sense to enforce diversity in the beginning, to better explore the possibilities
# if DIVERSE_VEHICLES is True, it enforces to consider all types of vehicles
# else it enforces all customers to be considered a starting point during the batch
ENFORCE_DIVERSE_START = [0,  0.6, 0.9]

# to force the algorithm to consider all types of vehicles for the first fraction of iterations
DIVERSE_VEHICLES = [True, False]

# proportion of vehicles to be kept
VEHICLE_PROPORTION = [0, 0.2]

tic = time.clock()

def run_aco(parameters):

    # how many good vehicle assingments of current batch
    # to reuse in the next batch
    # --> good to have a fraction of BATCH_SIZE
    # this way some good solutions are kept, but it is still possible to explore new combinations
    # it seems more reasonable that good vehicle combinations can be successfully applied again, if the choice of the first node is free and not randomized through ENFORCE_DIVERSE_START
    vehicles_kept = int(np.floor(parameters[5] * parameters[2]))


    aco.initialize(problem_path='problem1/', initial_pheromone=parameters[0])

    best = aco.do_iterations(parameters[1], parameters[2], vehicles_kept, parameters[3], parameters[4], parameters[0], parameters[5])

    toc = time.clock()

   # print('it took so many seconds to compute: ', toc - tic)

    return [i[0], i[1], i[2], vehicles_kept, i[3], i[4], best]



best_solutions = []
count = 1
for i in itertools.product(INITIAL_PHEROMONE, ITERATIONS, BATCH_SIZE, ENFORCE_DIVERSE_START, DIVERSE_VEHICLES, VEHICLE_PROPORTION):
    print(count)
    print(BATCH_SIZE)
    count += 1
    best_solutions.append(run_aco(i))
    #aco.do_iterations(iterations=ITERATIONS,
#                   batch_size=BATCH_SIZE,
#                   keep_v=KEEP_VEHICLES,
#                   enforce_diverse_start=ENFORCE_DIVERSE_START,
#                   all_vehicles=DIVERSE_VEHICLES)

date = time.time()

np.savetxt('results'+ str(date) + '.txt', best_solutions)
