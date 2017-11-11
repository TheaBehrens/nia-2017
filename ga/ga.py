import random
import numpy as np
import itertools, functools, operator
from functools import partial
from helpers import *

#initialize: (int, () -> chromosome) -> [chromosome]
def initalize(n, gen_chromosome):
	return [gen_chromosome() for i in range(n)]

#select: [chromosome] -> [chromosome]
def select_roulette(chromosomes, n, fitness_func):
	weights = [1.0 / fitness_func(x) for x in chromosomes]
	pdf = weights / sum(weights)
	selected = np.random.choice(len(chromosomes), size=n, replace=False, p=pdf)
	return np.take(chromosomes, selected, axis=0)

def select_nbest(chromosomes, n, fitness_func):
	return sorted(chromosomes, key=fitness_func)[-n:]

#recombine: chromosome, chromosome -> chromosome
def recombine_crossover(a, b, num_splits):
	split_indices = np.array(sorted(np.random.choice(np.arange(1, len(a)), size=num_splits, replace=False)))
	sa, sb = np.split(a, split_indices), np.split(b, split_indices)
	sa[0::2], sb[0::2] = sb[0::2], sa[0::2]
	sa, sb = np.concatenate(sa), np.concatenate(sb)
	return sa

#mutation: chromosome -> chromosome
def mutate_random(chromosome, alleles, p=0.1):
	r = random.random()
	if r < p:
		i = random.randint(0, len(chromosome) - 1)
		chromosome[i] = random.choice(alleles)	#random.randint(0, num_machines-1)
	return chromosome

def mutate_each_random(chromosome, alleles, p=0.05):
	for i in range(len(chromosome)):
		r = random.random()
		if r < p:
			chromosome[i] = random.choice(alleles)	#random.randint(0, num_machines-1)
	return chromosome

#replace: [chromosome], [chromosome] -> [chromsome]
def replace_all(offspring, parents):
	return offspring

def replace_keep_best(offspring, population, n, fitness_func):
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
	while i < max_generations:
		population = evolution_step(population, selection_func, crossover_func, mutation_func, replacement_func, fitness_func)
		print_stats(population, fitness_func)
		i += 1

def print_stats(population, fitness_func):
	fs = [fitness_func(p) for p in population]
	mean = np.mean(fs)
	max_f = np.max(fs)
	print("mean: {:.2f}, best: {:.2f}".format(mean, max_f))