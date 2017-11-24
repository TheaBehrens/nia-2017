import csv
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

# several functions have to know the distance and the pheromone of each path
distances = None
pheromone_mat = None

# some parameters...
INITIAL_PHEROMONE = 0.1
ants = 20
iterations = 101
evaporation_rate = 0.4
which_problem = 2
Q = 100
alpha=3
beta=3
savename = '24nov_' +str(ants)+ 'ants_' +str(iterations)+ 'iterations_' +str(evaporation_rate) +'roh_' +str(Q)+ 'q_' +str(alpha) + 'a_' +str(beta) + 'b' 

# helper function to read in the problem from file
def read_tsp(problem):
    tsp = []
    with open(str(problem) + '.tsp') as inputfile:
        for line in csv.reader(inputfile, delimiter=' '):
            line.remove('')
            tsp.append([int(i) for i in line])
    return tsp


# initialize:
# each arc has pheromone trail associated with it 
# --> same value for all arcs in the beginning
def initialize(problem_name):
    global distances, pheromone_mat
    dist = read_tsp(problem_name)
    distances = np.asarray(dist, dtype=int)
    # initializing all pheromone values to 0.1? good size? too high/low?
    pheromone_mat = np.ones(distances.shape) * INITIAL_PHEROMONE


# crawl / find tours
# each ant: find solution (crawl along a path)
# --> using the pheromone and desirablility
# probabilistic path construction: choose a path with probability of 
# pheromone^(alpha) * desirability^(beta) / (sum over these for all outgoing edges)
def solution_generation(nr_ants, alpha=1, beta=1):
    path_len = distances.shape[0]
    # solutions matrix
    solutions = np.zeros((nr_ants, path_len), dtype=int)
    # let all ants start at different points:
    solutions[:,0] = np.random.choice(path_len, nr_ants, replace=False)
    # solutions[:,0] = np.random.multinomial(path_len, [1/nr_ants]*nr_ants) # TODO how is this supposed to work?!
    print(solutions[:,0])
    # for each ant find a path through the graph
    for ant in range(nr_ants):
        not_visited = np.arange(path_len).tolist()
        for node in range(1, path_len):
            position = solutions[ant, node-1]
            not_visited.remove(position)
            # choose next edge
            solutions[ant, node] = choose_edge(position, not_visited, alpha, beta)
    return solutions

# of all remaining nodes, chose one based on the pheromone and the desirability
def choose_edge(position, open_nodes, alpha, beta):
    pheromone = np.power(pheromone_mat[position, open_nodes], alpha)
    desirability = np.power(1 / distances[position, open_nodes], beta)
    denominator = np.sum(pheromone * desirability)
    probabilities = pheromone * desirability / denominator
    chosen = np.random.choice(open_nodes, p=probabilities)
    return chosen

# calculate the cost of a path by summing over all edges
def path_cost(path):
    cost = 0
    for i in range(len(path)):
        cost += distances[path[i-1],path[i]]
    return(cost)

# add pheromone to all edges contained in a path
# amount of pheromone inversly proportional to the cost of that path
# parameter Q >= 1  can increase amout of pheromone 
def add_pheromone(path, cost, Q=1):
    global pheromone_mat
    additional_pheromone = Q/cost
    for i in range(len(path)):
        pheromone_mat[path[i-1],path[i]] += additional_pheromone


def pheromone_changes(solutions, evaporation_rate=0.01, Q=1):
    # evaporation:
    global pheromone_mat
    pheromone_mat = (1-evaporation_rate) * pheromone_mat

    # intensification:
    nr_ants = solutions.shape[0]
    cost_mat = np.zeros((nr_ants,))
    for ant in range(nr_ants):
        # calculate cost of path
        cost = path_cost(solutions[ant,:])
        cost_mat[ant] = cost
        # add pheromone to all edges included in path
        add_pheromone(solutions[ant,:], cost, Q)
    return cost_mat

# main stuff... can be in a loop later or in other file?
initialize(which_problem)

cost_in_iteration = np.zeros((iterations, 3))
tic = time.clock()
for i in range(iterations):
    solutions = solution_generation(ants, alpha=alpha, beta=beta)
    cost_mat = pheromone_changes(solutions, evaporation_rate=evaporation_rate, Q=Q)
    cost_in_iteration[i,:] = [np.mean(cost_mat), np.max(cost_mat), np.min(cost_mat)]

toc = time.clock()

print(toc - tic)

with open(savename+'.pickle', 'wb') as f:
    pickle.dump(cost_in_iteration, f, pickle.HIGHEST_PROTOCOL)

fig = plt.figure()
plt.plot(cost_in_iteration[:,0])
fig.show()
fig.savefig(savename + '.pdf')

