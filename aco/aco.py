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
        probs = generate_transition_probabilities(cost_matrix,
                                                  pheromones, alpha, beta)
        trails = generate_new_trails(num_ants, probs)
        pheromones = evaporate_trails(pheromones, rho)
        pheromones = intensify_trails(cost_matrix, trails, Q, pheromones)
        print(best_trail(cost_matrix, trails))

    return best_trail(cost_matrix, trails)


def trail_cost(cost_matrix, trail):
    cost = 0
    for node1, node2 in zip(trail, trail[1:]):
        cost += cost_matrix[int(node1), int(node2)]
    return cost


def best_trail(cost_matrix, trails):
    cost = np.max(cost_matrix) * np.size(cost_matrix, 0)
    for trail in trails:
        cost = min(trail_cost(cost_matrix, trail), cost)
    return cost


def generate_random_trails(num_ants, num_nodes):
    '''Generates valid random complete traversals for each ant starting at
    the same origin. This should only be used for the initial solution
    creation. Returns array of trails.

    '''
    trails = [[0] + random.sample(range(1, num_nodes), num_nodes - 1) for n in
              range(num_ants)]

    return np.array(trails)


def generate_new_trails(num_ants, probabilities):

    '''Takes as inputs num_ants and transition probabilities. Returns new
    trails based on these probabilities.

    '''
    num_nodes = np.size(probabilities, 0)
    trails = np.zeros((num_ants, num_nodes))
    for trail in trails:
        for node in range(1, len(trail)):
            while trail[node] == 0:
                for next_node in range(len(probabilities[node])):
                    if random.random() < probabilities[node][next_node]:
                        trail[node] = next_node
    return trails


def generate_transition_probabilities(cost_matrix, pheromones, alpha, beta):
    '''Takes as input the cost_matrix, pheromones, alpha and beta and
    determines transition probabilities.

    '''
    num_nodes = np.size(cost_matrix, 0)
    probabilities = np.zeros((num_nodes, num_nodes))

    for node1 in range(num_nodes):
        # first generate un-normalized probabilities
        for node2 in range(num_nodes):
            if cost_matrix[node1][node2] != 0:
                probabilities[node1][node2] = (pheromones[node1]
                                               [node2]**alpha *
                                               (1/cost_matrix[node1]
                                                [node2])**beta)
            # then normalize
    normalized = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            normalized[i][j] = \
                               np.divide(probabilities[i][j],
                                         np.sum(probabilities[i]))

    return normalized


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
            n1 = trail[i]
            if n1 == trail[-1]:  # check for wrap around
                n2 = trail[0]
            else:
                n2 = trail[i+1]
            pheromones[int(n1)][int(n2)] += Q / cost_matrix[int(n1)][int(n2)]

    return pheromones
