import matplotlib.pyplot as plt
import numpy as np
# where to take the fitness function from:
import fit_polynomial as fit

# algorithm parameters
NUM_PARAMS = len(fit.get_bounds())
POPULATION_SIZE = 10 * NUM_PARAMS
NUM_ITERATIONS = 300
CR = 0.9
F = 0.6

def initialize(pop_size):
    bounds = fit.get_bounds()
    pop = np.zeros((pop_size, len(bounds)))
    for i in range(pop_size):
        for j in range(len(bounds)):
            pop[i,j] = np.random.randint(bounds[j,0],bounds[j,1])
    return pop


def de_step(pop):
    new_pop = np.copy(pop)
    for i, x in enumerate(pop):
        u,v,w = pop[np.random.choice(len(pop), size=3, replace=False)]
        while u is x or u is x or u is x:
            u,v,w = pop[np.random.choice(len(pop), size=3, replace=False)]
        r = np.random.randint(0, NUM_PARAMS)
        p = np.random.rand(NUM_PARAMS) < CR
        p[r] = True
        z = u + F*(v-w)
        y = np.where(p, z, x)
        if fit.fitness(y) < fit.fitness(x):
            new_pop[i] = y
    return new_pop


population = initialize(POPULATION_SIZE)

figure, ax1 = plt.subplots()

plt.ion() # interactive!
x_range = fit.get_x()
points = fit.get_points()

for i in range(NUM_ITERATIONS):
    population = de_step(population)
    # the best individual of the population:
    best_fitness = 10000
    best_x = 0
    for _, x in enumerate(population):
        if(fit.fitness(x) < best_fitness):
            best_fitness = fit.fitness(x)
            best_x = x
            
    # plotting and printing progress:
    if(i%2 == 0):
        print(i)
        #print('%.2f , %.2f, value: %.2f' %(best_x[0], best_x[1], best_fitness))
        ax1.cla() # clear previous lines
        ax1.scatter(x_range, points)
        ax1.set_ylim(-2,2)
        for i, x in enumerate(population):
            ax1.plot(x_range, fit.curve(x), 'grey')
        ax1.plot(x_range, fit.curve(best_x), 'r')

        plt.pause(0.01)


# just so that the figure does not go away directly
# gives you the chance to look at it for a moment
plt.pause(5)
