from ga import *
import imp

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

jobs, num_machines = load_data("makespan_1.dzn")  # need to change here for each problem

# problem specific functions
def gen_random_chromosome(num_jobs, num_machines):
	return [random.randint(0, num_machines-1) for i in range(len(jobs))]

def fitness(chromosome, job_times):
	time_per_machine = np.zeros(num_machines)
	for job_index, machine_index in enumerate(chromosome):
		time_per_machine[machine_index] += job_times[job_index]
	worst_time = np.max(time_per_machine)
	return -worst_time

# run things!
gen_chromosome = lambda: gen_random_chromosome(len(jobs), num_machines)
alleles = list(range(num_machines))

evolve(
	fitness_func = partial(fitness, job_times = jobs),
	init_func = partial(initalize, n = population_size, gen_chromosome = gen_chromosome),
	selection_func = partial(select_roulette, n = 25),
	crossover_func = partial(recombine_crossover, num_splits = 3),
	mutation_func = partial(mutate_each_random, p = 0.01, alleles=alleles),
	replacement_func = partial(replace_keep_best, n = 5),
	max_generations = 2000)

limit = np.sum(jobs) / num_machines
print("lower limit: {}".format(limit))


