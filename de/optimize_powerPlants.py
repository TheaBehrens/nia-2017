import matplotlib.pyplot as plt
import numpy as np
# where the problem to optimize is defined:
import powerplants_noa as fit

# algorithm parameters
NUM_PARAMS = len(fit.get_bounds())
POPULATION_SIZE = 5 * NUM_PARAMS
NUM_ITERATIONS = 700
CR = 0.9
F = 0.7

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
        # some problem specific constraint:
        # reasonable prices are in the range of 0-0.5
        # if they leave that interval --> reinitialize!
        # in this case HIGHER values are better:
        while ((y[6] > 0.5)):
            y[6] = np.random.rand()
        while ((y[7] > 0.5)):
            y[7] = np.random.rand()
        while ((y[8] > 0.5)):
            y[8] = np.random.rand()
        if fit.objective_func(y) > fit.objective_func(x):
            new_pop[i] = y
    return new_pop


population = initialize(POPULATION_SIZE)
# storing best value of each iteration to plot it
performance = np.zeros((NUM_ITERATIONS,1))
best_results = np.zeros((NUM_ITERATIONS,9))


for i in range(NUM_ITERATIONS):
    population = de_step(population)
    # the best individual of the population:
    if(i%10 == 0):
        best_objective = -np.inf
        best_x = 0
        for _, x in enumerate(population):
            if(fit.objective_func(x) > best_objective):
                best_objective = fit.objective_func(x)
                best_x = x
        # to watch how the values get better:
        print(best_objective)
    performance[i] = best_objective
    best_results[i,:] = best_x

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

plt.figure()
plt.plot(range(NUM_ITERATIONS), performance)
plt.show()


# would be even better if not considering fixed bounds for the colors,
# but instead take all with negative revenues to be red,
# all with revenues in the range of 0-1,100,000 to be yellow 
# and the rest to be green or so...

# and if you would plot it during the search, you could see if it is on a promissing track or not and abort search if it's not reasonable


# quite often the third plot shows prices in the hundreds, instead of something below 0
# in these cases also the amount to be sold there is close to 0, but that does not seem like a clever strategy
# --> added a constraint now, such that the prices have to be smaller than 0.5
# --> leads to more reliable convergence to good solutions
fig, axes = plt.subplots(3,1, sharex=True)
axes[0].scatter(best_results[:100,6], best_results[:100,3], color='r')
axes[0].scatter(best_results[100:300,6], best_results[100:300,3], color='y')
axes[0].scatter(best_results[300:,6], best_results[300:,3], color='g')

axes[1].scatter(best_results[:100,7], best_results[:100,4], color='r')
axes[1].scatter(best_results[100:300,7], best_results[100:300,4], color='y')
axes[1].scatter(best_results[300:,7], best_results[300:,4], color='g')

axes[2].scatter(best_results[:100,8], best_results[:100,5], color='r')
axes[2].scatter(best_results[100:300,8], best_results[100:300,5], color='y')
axes[2].scatter(best_results[300:,8], best_results[300:,5], color='g')

plt.show()

