from ga import *
import imp
import pickle
from time import time

# * other than random initalize
# * fancy crossover, only have n-splits at the moment
# * other selection method (tournament?)
# * loading job description from text files
# * evaluation of different parameters, method, etc

# general parameters
population_size = 150

# problem definition and loading, load data files here
def load_data(filename):
        f = open(filename)
        data = imp.load_source('data', '', f)
        f.close()
        return data.jobs, data.num_machines

jobs, num_machines = load_data("makespan_2.dzn")  # need to change here for each problem

# problem specific functions
def gen_random_chromosome(num_jobs, num_machines):
	return [random.randint(0, num_machines-1) for i in range(len(jobs))]

def fitness(chromosome, job_times):
	time_per_machine = np.zeros(num_machines)
	for job_index, machine_index in enumerate(chromosome):
		time_per_machine[machine_index] += job_times[job_index]
	worst_time = np.max(time_per_machine)
	return -worst_time

def function_grid(param_grid):

    gen_chromosome = lambda: gen_random_chromosome(len(jobs), num_machines)
    alleles = list(range(num_machines))
    count = 0
    num_combinations = 8
    max_generations = 2000
    stats_array = np.zeros((max_generations, 2, num_combinations))
    time_array = np.zeros((num_combinations,))
    # change to two if new functions implemented
    for i in range(1):
        for j in range(2):
            for k in range(1):
                for l in range(2):
                    for m in range(2):
                        start = time()
                        temp_stats = evolve(
                                	fitness_func = partial(fitness, job_times = jobs),
                                	init_func = partial(initalize, n = population_size, gen_chromosome = gen_chromosome,choice = param_grid[0][i]),
                                	selection_func = partial(select, n = 25,choice = param_grid[1][j]),
                                	crossover_func = partial(recombine, num_splits = 3, choice = param_grid[2][k]),
                                	mutation_func = partial(mutate, p = 0.01, alleles=alleles, choice = param_grid[3][l]),
                                	replacement_func = partial(replace, n = 5, choice = param_grid[4][m]),
                                	max_generations = max_generations)
                        time_array[count] = time()-start
                        limit = np.sum(jobs) / num_machines
                        stats_array[:, :, count] = temp_stats
                        print('Combination {} done'.format(count))

                        count = count + 1
    with open('makespan_2_results.pickle','wb') as f:
        pickle.dump(stats_array,f,pickle.HIGHEST_PROTOCOL)
                            
    with open('makespan_2_times.pickle','wb') as f:
        pickle.dump(time_array,f,pickle.HIGHEST_PROTOCOL)
            
    

# run things!


param_grid = [['random','random'],['roulette','n_best'],
              ['crossover','crossover'],['random','each_random'], 
              ['all','keep_best']]

function_grid(param_grid)
#stats_array = evolve(
#	fitness_func = partial(fitness, job_times = jobs),
#	init_func = partial(initalize, n = population_size, gen_chromosome = gen_chromosome),
#	selection_func = partial(select_roulette, n = 25),
#	crossover_func = partial(recombine_crossover, num_splits = 3),
#	mutation_func = partial(mutate_each_random, p = 0.01, alleles=alleles),
#	replacement_func = partial(replace_keep_best, n = 5),
#	max_generations = 2000)

#with open('test1.pickle','wb') as f:
#    pickle.dump(stats_array,f,pickle.HIGHEST_PROTOCOL)
#
#print("lower limit: {}".format(limit))
#
#
