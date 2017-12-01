import numpy as np
import pickle

factor = 0.5
# linear separable or not: <0.2 if separable >0.9
crossover_rate = 0.8
population_size = 20
# smarter? better read in
bounds = np.array([[-100,100],[-100,100],[-100,100],[-100,100],[-100,100]])

def problem_func(x):
    return sum(x*x)

def initialize(population_size, bounds):
    population = np.zeros((population_size, len(bounds)))
    for i in range(population_size):
        for j in range(len(bounds)):
            population[i,j] = np.random.randint(bounds[j,0],bounds[j,1])

    return population

def mutate(population, factor, donor):
    # look after donor value

    a,b,c = np.random.choice(len(population), 3, replace = False)
    donor_vector = population[a] + factor * (population[b]-population[c])
    return donor_vector

def crossover(donor, target, crossover_rate):
    mask = np.random.rand(donor.size) <= crossover_rate
    trial = target.copy()
    trial[mask] = donor[mask]
    return trial

def select(trial, target):
    if problem_func(trial) <= problem_func(target):

        return trial

    else:
        return target


def evolve(population, factor, crossover_rate):
    # stopping criterion while TODO
    generation = 0
    stop = False
    pop_evolved = population.copy()
    while not stop:
        generation += 1
        if generation > 200:
            stop = True
        for i in range(population_size):

            donor_vector = mutate(population, factor, i)
            #print("Donor " , donor_vector)
            trial_vector = crossover(donor_vector, population[i, :], crossover_rate)
            #print("Trial ", trial_vector)
            evolved_vector = select(trial_vector, population[i, :])
        #    print("Evolved ", evolved_vector)
            pop_evolved[i, :] = evolved_vector
        population = pop_evolved
        print(np.mean(problem_func(population)))
    return pop_evolved


population = initialize(population_size,bounds)
evolved_population = evolve(population, factor, crossover_rate)
