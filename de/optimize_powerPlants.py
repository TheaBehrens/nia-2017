import matplotlib.pyplot as plt
import numpy as np
# where the problem to optimize is defined:
import powerplants_noa as fit

# algorithm parameters
NUM_PARAMS = len(fit.get_bounds())
POPULATION_SIZE = 5 * NUM_PARAMS
NUM_ITERATIONS = 500
CR = 0.4
F = 0.6

def initialize(pop_size):
    bounds = fit.get_bounds()
    pop = np.zeros((pop_size, len(bounds)))
    for i in range(pop_size):
        for j in range(len(bounds)):
            pop[i,j] = bounds[j,0] + np.random.rand() * (bounds[j,1] - bounds[j,0])
    return pop


def de_step(pop):
    new_pop = np.copy(pop)
    for i, x in enumerate(pop):
        u,v,w = pop[np.random.choice(len(pop), size=3, replace=False)]
        while u is x or v is x or w is x:
            u,v,w = pop[np.random.choice(len(pop), size=3, replace=False)]
        r = np.random.randint(0, NUM_PARAMS)
        p = np.random.rand(NUM_PARAMS) < CR
        p[r] = True
        z = u + F*(v-w)
        y = np.where(p, z, x)
        # in this case HIGHER values are better:
        if fit.objective_func(y) > fit.objective_func(x):
            new_pop[i] = y
    return new_pop


population = initialize(POPULATION_SIZE)


for i in range(NUM_ITERATIONS):
    population = de_step(population)
    # the best individual of the population:
    best_objective = -np.inf
    best_x = 0
    for _, x in enumerate(population):
        if(fit.objective_func(x) > best_objective):
            best_objective = fit.objective_func(x)
            best_x = x
    # to watch how the values get better:
    print(best_objective)

# in the end, to print the best solution that was found:
for _, x in enumerate(population):
    best_objecive = np.inf
    best_x = 0
    if(fit.objective_func(x) < best_objecive):
        best_objective = fit.objective_func(x)
        best_x = x
print(best_x)

# # sanity check: does it give correct result for a solution?
# --> yes, is 200,000
# hand_assigned = np.array([0,0,4000000,1000000,3000000,0,0.225,0.125,1])
# print(fit.objective_func(hand_assigned))

