import csv
import numpy as np

# several functions have to know the distance and the pheromone of each path
distances = None
pheromone_mat = None
capacity = None
demand = None
t_cost = None

# TODO ideas to add:
# different pheromones for different vehicles
# is there any other way to learn good vehicle assignments? at the moment always only random
# 
# Did incorporate the demand vs left stock now: but that slows down and does not seem to improve results, so I commented it out again for now.

# helper function to read in the problem from file
def read_file(filename):
    tsp = []
    with open(str(filename)) as inputfile:
        for line in csv.reader(inputfile, delimiter=' '):
            line.remove('')
            tsp.append([int(i) for i in line])
    return tsp


# initialize:
# each arc has pheromone trail associated with it
# --> same value for all arcs in the beginning
def initialize(problem_path, initial_pheromone):
    global distances, pheromone_mat, capacity, demand, t_cost
    dist = read_file(problem_path + 'distance.txt') 
    capacity = read_file(problem_path + 'capacity.txt')
    demand = read_file(problem_path + 'demand.txt')
    t_cost = read_file(problem_path + 'transportation_cost.txt')

    capacity = np.asarray(capacity, dtype=int).squeeze() # (33,)
    demand = np.asarray(demand, dtype=int).squeeze()     # (100,)
    # add a demand of 0 for the depot, in order to be able 
    # to use the same indices for demand and distances
    demand = np.insert(demand, 0, 0)                     # (101,)
    distances = np.asarray(dist, dtype=int)              # (101,101)
    t_cost = np.asarray(t_cost, dtype=int).squeeze()     # (33,)
    print(capacity)
    print(t_cost)
    
    # initializing all pheromone values to the specified value
    pheromone_mat = np.ones(distances.shape) * initial_pheromone


# crawl / find tours
# each ant: find solution (crawl along a path)
# --> using the pheromone and desirablility AND DEMAND AND CAPACITY
# probabilistic path construction: choose a path with probability of
# pheromone^(alpha) * desirability^(beta) * demand_fit^(gamma) / 
#               (sum over these for all outgoing edges)
def solution_generation(start=None, alpha=1, beta=1, gamma=1):
    first = False
    if (start):
        first = True
    open_demand = np.copy(demand)
    # open_demand = open_demand[:10]
    
    # solution stored as a list of lists
    # a list for each used vehicle: 
    #       with the customer_ids in the order of visiting
    solutions = []

    # Take a random vehicle and let it serve customers until empty, 
    # decrease the demand of the served customers on the way
    # and also decrease left stock (to know when empty!)
    # Then take next vehicle and repeat, until no customer has open demand left

    # in order to minimize shuffle calls, get a list of indices, 
    # in which order to deploy the vehicles
    indices = np.arange(len(t_cost))
    np.random.shuffle(indices)
# the following lines are of course very problem specific, would need to make it more general, to include all types of vehilces into the solution...
# it does increase the probability of good solutions by quite a margin though
    important_indices = np.array([0,-1, 20, -3])
    np.random.shuffle(important_indices)
    np.random.shuffle(indices)
    indices = np.concatenate([important_indices, indices])
    

    # continue until all customers are served
    i = 0
    while(sum(open_demand) != 0):
        # take the first vehicle and let it find serve customers until empty
        left_stock = capacity[indices[i]]
        route = [0] # start at depot
        while(left_stock > 0):
             # the current position of the ant is written at the last position of route
             next_customer = choose_customer(route[-1], open_demand, left_stock)
             if first:
                 next_customer = start
                 first = False
             if(open_demand[next_customer] <= left_stock):
                 left_stock -= open_demand[next_customer]
                 open_demand[next_customer] = 0
             else:
                 open_demand[next_customer] -= left_stock
                 left_stock = 0
             route.append(next_customer) 
             if(sum(open_demand)==0):
                 break
        # this ant finished, let the next take over
        i += 1
        solutions.append(route)
    return (solutions, indices)

# of all remaining nodes, chose one based on 
#   - the pheromone,
#   - the distance,
#   - demand/stock
def choose_customer(position, open_demand, stock):
    interesting_idx = open_demand > 0
    pheromone = pheromone_mat[position, interesting_idx]
    closeness = 1 / distances[position, interesting_idx]
    '''
    # TODO: also incorporate the demand here
    d = open_demand[interesting_idx]
    d = d-stock # if something is zero:  BEST
                # if it is negative: okay (we can satisfy the demand here, closer to zero: better)
                # something positive: avoid, we can not serve as much as demanded
    demand_fits = np.ones(d.shape)
    demand_fits[d<0] = -d[d<0]
    demand_fits[d==0] =  max(demand_fits) * 4 # 4 times the next best value
    '''
    denominator = np.sum(pheromone * closeness)
    probabilities = pheromone * closeness / denominator

    chosen = np.random.multinomial(1, probabilities).argmax()
    true_idx = np.where(open_demand > 0)[0][chosen]
    return true_idx

# calculate the cost of a path by summing over the costs of all tours
def path_cost(tours, vehicle_idx):
    cost = 0
    # for each tour
    for i in range(len(tours)):
        fuel_consumption = t_cost[vehicle_idx[i]]
        tour_length = 0
        # go through all nodes in the tour to calculate the lengh of it:
        for j in range(len(tours[i])):
            if j == 0:
                # also adding the way back from last customer to depot
                tour_length += distances[tours[i][-1],0]
            else:
                tour_length += distances[tours[i][j-1], tours[i][j]]

        cost += fuel_consumption * tour_length
    return(cost)

# add pheromone to all edges contained in a path
# amount of pheromone inversly proportional to the cost of that path
# parameter Q >= 1  can increase amout of pheromone
def add_pheromone(tours, cost, Q):
    global pheromone_mat
    additional_pheromone = Q/cost
    # print(additional_pheromone)
    for i in range(len(tours)):
        # go through all nodes in the tour to calculate the lengh of it:
        for j in range(len(tours[i])):
            if j == 0:
                # also adding pheromone on the way from last customer to depot
                pheromone_mat[tours[i][-1],0] += additional_pheromone
            else:
                pheromone_mat[tours[i][j-1], tours[i][j]] += additional_pheromone

# evaporation and intensification
def pheromone_changes(tours, vehicle_idx, evaporation_rate=0.05, Q=1000):
    # evaporation:
    global pheromone_mat
    pheromone_mat = (1-evaporation_rate) * pheromone_mat

    # intensification:
    nr_ants = 1 # TODO: several ants?
    cost_mat = np.zeros((nr_ants,))
    for ant in range(nr_ants):
        # calculate cost of path
        cost = path_cost(tours, vehicle_idx)
        cost_mat[ant] = cost
        # add pheromone to all edges included in path
        add_pheromone(tours, cost, Q)
    return cost_mat


# yeah, well
# you enforce diversity with this, but that does also enforce to start with bad nodes in quite a few cases, so solutions overall are worse
# maybe the MIN is better? 
def collect_several_solutions():
    # maybe don't update results directly, but let solution paths start at each of the cities/customers
    # --> would avoid bias towards taking the nearby cities always first
    # also there are several pheromone trails, and not only one is updated...
    sol_list = []
    total_cost = 0
    for i in range(1,distances.shape[0]):
        # with an 'i' in the call solution_generation(i)
        # --> force the system to consider all nodes as starting points
        sol_list.append(solution_generation(i))
    for i in range(len(sol_list)):
        cost = pheromone_changes(sol_list[i][0], sol_list[i][1])
        if (cost[0] < 90000):
            print(cost[0])
        total_cost += cost
    # not sure how much fluktuation in the solutions there is based on first initialization or so
# but solutions seem to be quite a lot better on average with this 'batch updating' instead of updating after every iteration
# (107.000 instead of 110.000)
    print(total_cost/len(sol_list))
    
# current best: 82328 with all start nodes, and demand_fit
#    79113 with demand fit, without all start nodes
#    78546 without demand fit, and without all start, also only 19 sec instead of 29
#    78156 without demand fit, with all start

def do_iterations(iter_nr):
    total_cost = 0
    for i in range(iter_nr):
        ## (tours, idx) = solution_generation()
        ## cost = pheromone_changes(tours, idx)
        ## total_cost += cost
        ## if i % 100 == 0:
        ##     print(total_cost/100)
        ##     total_cost = 0
        collect_several_solutions()
        '''
        if i % 10 == 0:
            print(cost_mat)
        if i % 500 == 0:
            # just to get some feeling for the values in the pheromone_mat
            print(np.mean(pheromone_mat))
            print(np.max(pheromone_mat))
            print(np.min(pheromone_mat))
            '''
