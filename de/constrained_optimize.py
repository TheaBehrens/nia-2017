import matplotlib.pyplot as plt
import numpy as np
# where to take the fitness function from:
import constrained_problem as fit

# algorithm parameters
POPULATION_SIZE = 30
MIN, MAX = -5, 5
NUM_PARAMS = 2
NUM_ITERATIONS = 91
CR = 0.1
F = 0.1

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
        if fit.fitness(y) < fit.fitness(x):
            new_pop[i] = y
    return new_pop


population = np.random.rand(POPULATION_SIZE, NUM_PARAMS) * (MAX-MIN) + MIN
tau = np.linspace(0, 20, 200)

plt.ion() # interactive!
fig, (ax1, ax2, ax3) = plt.subplots(3,1)
color = 'gbcmy'
cnt = 0
ax3.set_xlim(0, NUM_ITERATIONS)
ax3.set_ylim(-1,4)

for i in range(NUM_ITERATIONS):
    population = de_step(population)
    # the best individual of the population:
    best_fitness = 100000
    best_individual = 0
    for _, x in enumerate(population):
        if(fit.fitness(x) < best_fitness):
            best_fitness = fit.fitness(x)
            best_x = x
            
    if(i%5 == 0):
        mean_fitness = 0
        print('%.2f , %.2f, value: %.2f' %(best_x[0], best_x[1], best_fitness))
        ax1.cla() # clear previous lines
        for _, x in enumerate(population):
            ax1.plot(tau, fit.func_h(x[0], x[1], tau), 'grey')
            ax2.scatter(x[0], x[1], c=color[cnt%4])
            mean_fitness += fit.fitness(x)
        ax2.scatter(best_x[0], best_x[1], c='r')
        ax3.plot(i, mean_fitness/len(population), 'b^')
        ax3.plot(i, best_fitness, 'ro')
        cnt += 1
        ax1.plot(tau, fit.func_h(best_x[0], best_x[1], tau), 'r')
        ax1.fill_between(tau[tau<10], 1.04, 2, alpha=0.3)
        ax1.fill_between(tau[tau>=10], 0.8, 2, alpha=0.3)
        ax1.fill_between(tau[tau<5], 0.4, -1, alpha=0.3)
        ax1.set_ylim(-0.5,1.5)

        plt.pause(0.05)


# just so that the figure does not go away directly
# gives you the chance to look at it for a moment
plt.pause(5)
