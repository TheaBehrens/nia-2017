from ga import *
import imp
import itertools
import sys
import pickle

# load problem
def load_data(filename):
        f = open(filename)
        data = imp.load_source('data', '', f)
        f.close()
        return data.jobs, data.num_machines

jobs, num_machines = load_data("makespan_1.dzn")  # need to change here for each problem

def gen_random_chromosome(num_jobs, num_machines):
	return [random.randint(0, num_machines-1) for i in range(len(jobs))]

def gen_chromosome():
	return gen_random_chromosome(len(jobs), num_machines)

def fitness(chromosome, job_times):
	time_per_machine = np.zeros(num_machines)
	for job_index, machine_index in enumerate(chromosome):
		time_per_machine[machine_index] += job_times[job_index]
	worst_time = np.max(time_per_machine)
	return -worst_time

alleles = list(range(num_machines))
fitness_func = partial(fitness, job_times = jobs)

# parameters
population_size = 200
num_trials = 6
num_generations = 2000

param_values = [
	# init
	[partial(initalize, n = population_size, gen_chromosome = gen_chromosome)],
	# selection
	[partial(select_tournament, tournament_size = 10, n = 35),
	 partial(select_roulette, n = 35)],
	# crossover
	[partial(recombine_crossover, num_splits = 1),
	 partial(recombine_crossover, num_splits = 4)],
	# mutation
	[partial(mutate_random, p = 0.1, alleles=alleles),
	 partial(mutate_each_random, p = 0.01, alleles=alleles)],
	# replacement
	[partial(replace_all),
	 partial(replace_keep_best, n = 5, fitness_func=fitness_func)]
]

visible_params = ["num_splits", "n", "tournament_size", "p"]

def partial_to_info(partial_func):
	name = partial_func.func.__name__
	keywords = partial_func.keywords
	visible = { p: keywords[p] for p in visible_params if p in keywords }
	if len(visible) > 0:
		return name + " " + str(visible)
	else:
		return name

def evaluate_parameters(params):
	stats = []
	init_func, params = params[0], params[1:]
	for i in range(num_trials):
		stats.append([])
		population = init_func()
		for j in range(num_generations):
			population = evolution_step(population, fitness_func=fitness_func, *params)
			stats[i].append(collect_stats(population))
	desc = ", ".join(map(partial_to_info, params))
	return stats, desc

def collect_stats(population):
	fit = np.apply_along_axis(fitness_func, 1, population)
	return np.max(fit)

if __name__ == "__main__":
	import multiprocessing as mp
	pool = mp.Pool(4)
	parameters = list(itertools.product(*param_values))
	results = list(pool.map(evaluate_parameters, parameters))
	pickle.dump(results, open("results.pickle", "wb"))