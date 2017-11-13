import random
import numpy as np
import itertools, functools, operator
from functools import partial
from helpers import *


#initialize: (int, () -> chromosome) -> [chromosome]
def initalize(n, gen_chromosome, choice = 'random'):
    if choice =='random':
        return [gen_chromosome() for i in range(n)]
    # replace with different function
    else:
        return [gen_chromosome() for i in range(n)]

#select: [chromosome] -> [chromosome]
def select(chromosomes, n, fitness_func,choice='roulette'):
    if choice == 'roulette':
    	weights = [1.0 / fitness_func(x) for x in chromosomes]
    	pdf = weights / sum(weights)
    	selected = np.random.choice(len(chromosomes), size=n, replace=False, p=pdf)
    	return np.take(chromosomes, selected, axis=0)
    # n best function
    else:
        return sorted(chromosomes, key=fitness_func)[-n:]

#recombine: chromosome, chromosome -> chromosome
def recombine(a, b, num_splits,choice = 'crossover'):
    if choice == 'crossover':
    	split_indices = np.array(sorted(np.random.choice(np.arange(1, len(a)), size=num_splits, replace=False)))
    	sa, sb = np.split(a, split_indices), np.split(b, split_indices)
    	sa[0::2], sb[0::2] = sb[0::2], sa[0::2]
    	sa, sb = np.concatenate(sa), np.concatenate(sb)
    	return sa
    # replace with new function
    else:
        split_indices = np.array(sorted(np.random.choice(np.arange(1, len(a)), size=num_splits, replace=False)))
        sa, sb = np.split(a, split_indices), np.split(b, split_indices)
        sa[0::2], sb[0::2] = sb[0::2], sa[0::2]
        sa, sb = np.concatenate(sa), np.concatenate(sb)
        return sa
#mutation: chromosome -> chromosome
def mutate(chromosome, alleles, p=0.1,choice='random'):
    if choice == 'random':
        r = random.random()
        if r < p:
            i = random.randint(0, len(chromosome) - 1)
            chromosome[i] = random.choice(alleles)	#random.randint(0, num_machines-1)
        return chromosome
        #each random
    else:
        for i in range(len(chromosome)):
            r = random.random()
            if r < p:
                chromosome[i] = random.choice(alleles)	#random.randint(0, num_machines-1)
        return chromosome

	
#replace: [chromosome], [chromosome] -> [chromsome]

def replace(offspring, population, n, fitness_func, choice = 'all'):
    if choice == 'all':
        return offspring
    
    # keep best
    else:
    	best = sorted(population, key=fitness_func)[-n:]
    	kept_offspring = random.sample(offspring, len(population) - n)
    	return best + kept_offspring

#choose parents for crossover: [chromosome], int, (chromosome, chromosome -> chromosome) -> [chromosome]
def generate_offspring(parents, n, crossover_func):
    parent_pairs = gen_random_pairs(parents)
    offspring = [crossover_func(*p) for p in itertools.islice(parent_pairs, n)]
    return offspring

def evolution_step(population, selection_func, crossover_func, mutation_func, replacement_func, fitness_func):
    offspring_count = len(population)	# does this make sense? should be a parameter...
    xs = selection_func(population, fitness_func=fitness_func)
    xs = generate_offspring(xs, offspring_count, crossover_func)
    xs = list(map(mutation_func, xs))
    xs = replacement_func(xs, population, fitness_func=fitness_func)
    return xs

def evolve(init_func, selection_func, crossover_func, mutation_func, replacement_func, fitness_func, max_generations=10000):
    population = init_func()
    i = 0
    stats_array = np.zeros((max_generations,2))
    while i < max_generations:
        population = evolution_step(population, selection_func, crossover_func, mutation_func, replacement_func, fitness_func)
        stats_array[i,:] = print_stats(population, fitness_func)
        i += 1
    return stats_array

def print_stats(population, fitness_func):
    fs = [fitness_func(p) for p in population]
    mean = np.mean(fs)
    max_f = np.max(fs)
    #print("mean: {:.2f}, best: {:.2f}".format(mean, max_f))
    return [mean,max_f]
