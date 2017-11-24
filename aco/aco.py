import numpy as np
import random


def ant_colony_optimize_tsp(num_ants, cost_matrix, rho, Q, alpha,
                            beta, iters):
    '''Runs ant colony optimization on a tsp problem set.

    '''
    # initialize random trails for each ant
    # create pheromones on used paths
    # loop until iterations limit reached
    # 1. generate trail for each ant
    # 2. evaporate pheromones
    # 3. intensify trails
    num_nodes = np.size(cost_matrix, 1)
    trails = generate_random_trails(num_ants, num_nodes)
    pheromones = intensify_trails(cost_matrix, trails, Q)
    for i in range(0, iters):
        trails = generate_new_trails(num_ants, cost_matrix, trails,
                                     pheromones, alpha, beta)
        pheromones = evaporate_trails(pheromones, rho)
        pheromones = intensify_trails(cost_matrix, trails, Q, pheromones)


def generate_random_trails(num_ants, num_nodes):
    '''Generates valid random complete traversals for each ant starting at
    the same origin. This should only be used for the initial solution
    creation. Returns array of trails.

    '''
    trails = [[0] + random.sample(range(1, num_nodes), num_nodes - 1) for n in
              range(num_ants)]

    return np.array(trails)


def generate_new_trails(num_ants, cost_matrix, current_trails,
                        pheromones, alpha, beta):

    '''Takes as inputs num_ants, current_trails, pheromones, and heuristic
    parameters (alpha, beta). Generates a single trail for each ant.
    Returns an array of trails.

    '''
    raise NotImplementedError

    # TODO: make new trails based on probabilities based on pheromones
    # and path costs

    # for each ant, start at the origin then choose the next node
    # based on the probability (pheromones and cost)

    # return np.array(trails)


def evaporate_trails(pheromones, rho):
    '''Scales down the pheromones based on rho parameter.

    '''
    # uses the function (from wikipedia):
    # tau_xy <- (1 - rho)*tau_xy
    pheromones = np.multiply((1 - rho), pheromones)
    return pheromones

  
def intensify_trails(cost_matrix, trails, Q, pheromones=False):
    '''Update the trails based on pheromones deposited.  Q is a constant
    that determines how much pheromone is deposited on a path by an
    individual ant.

    '''
    # use the function (from wikipedia):
    # tau_xy <- evaporated_trails + sum_k(delta tau_xy^k)
    # delta tau_xy^k = Q/cost (if ant uses path xy), else 0

    num_nodes = np.size(cost_matrix, 0)

    if pheromones is False:  # this indicates it is the first generation
        #  then initialize an empty array
        pheromones = np.zeros((num_nodes, num_nodes))

    # if an ant uses an edge, a scaled pheromone is added
    for trail in trails:
        for i in range(len(trail)):
            if i == len(trail) - 1:
                n1 = trail[i]
                n2 = trail[0]
            else:
                n1, n2 = trail[i:i+2]
            pheromones[n1, n2] += Q / cost_matrix[n1, n2]

    return pheromones


