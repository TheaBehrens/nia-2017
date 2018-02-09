import csv
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt   
import helpers

# several functions have to know the distance and the pheromone of each path
distances = None
phero_mats = None
capacity = None
demand = None
t_cost = None

# TODO ideas to add:
# is there any other way to learn good vehicle assignments? 
# 
# maybe could also enforce vehicle-diversity?
# otherwise the random shuffeling will 'prefer' vehicles with small capacitiy, as they exist in greater numbers
# 4 different vehicle-types
# in this case solution could be
# 2x 1000
# 4x 500
# 7x 300
# 20x 100
# 1x 1000, 2x 500
# ...
# 

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
    global distances, phero_mats, capacity, demand, t_cost
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
    types = np.unique(capacity)
    
    # initializing all pheromone values to the specified value
    # assuming that the value changes in capacity and cost are at the same places
    # (== we have several different vehicles, but for one capacity, the fuel cost is always the same)
    # make as many pheromone matrices as we have different vehicles
    phero_mats = np.ones((len(types), distances.shape[0], distances.shape[1])) * initial_pheromone


# crawl / find tours
# each ant: find solution (crawl along a path)
# --> using the pheromone and desirablility AND DEMAND AND CAPACITY
# probabilistic path construction: choose a path with probability of
# pheromone * desirability * demand_fit / 
#               (sum over these for all outgoing edges)
# Not using alpha, beta and gamma at the moment to weight the different components
# just take them all to equal parts
def solution_generation(visit_first=None, vehicles=None, alpha=1, beta=1, gamma=1):

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
    if(vehicles is None):
        vehicles = np.arange(len(t_cost))
        np.random.shuffle(vehicles)
        # be sure to include the different types of vehicles... 
        # should be made more general, not so problem specific as it is at the moment...
        # important_vehicles = np.array([0,-1, 20, -3])
        # np.random.shuffle(important_vehicles)
        # vehicles = np.concatenate([important_vehicles, vehicles])

    # weight matrix can be precomputed for a whole iteration
    closeness = 1.0 / (distances + 1e-20)             # this NEVER changes, could be computed just once
    cost_mat = np.zeros_like(phero_mats)
    for i in range(phero_mats.shape[0]):
        temp = phero_mats[i, :, :]
        cost_mat[i, :, :] = temp * closeness    

    # continue until all customers are served
    i = 0
    while(sum(open_demand) != 0):
        # take the first vehicle and let it serve customers until empty
        left_stock = capacity[vehicles[i]]
        route = [0] # start at depot
        v_type = idx_2_type(vehicles[i])
        while(left_stock > 0):
            
            # the current position of the ant is written at the last position of route
            next_customer = choose_customer(route[-1], open_demand, left_stock, v_type, cost_mat)
            # if the first customer to visit was given as an argument, 
            # overwrite the customer found with the given one
            if visit_first:
                next_customer = visit_first
                visit_first = False
            # decrease the left stock and the demand of the served customer
            if(open_demand[next_customer] <= left_stock):
                left_stock -= open_demand[next_customer]
                open_demand[next_customer] = 0
            else:
                open_demand[next_customer] -= left_stock
                left_stock = 0
            route.append(next_customer) 
            if(sum(open_demand)==0):
                # all customers are served, no need to search further
                break
        # this ant finished, let the next take over
        i += 1
        solutions.append(route)
    return (solutions, vehicles)

# helper function
# to transform the index of a vehicle into the type
def idx_2_type(idx):
    types = np.unique(capacity)
    for i in range(len(types)):
        if capacity[idx] == types[i]:
            return i

# of all remaining nodes, chose one based on 
#   - the pheromone,
#   - the distance,
#   - demand/stock --> took that part out again, because it only slowed down and did not improve the solution
def choose_customer(position, open_demand, stock, v_type, cost_mat):
    interesting_idx = open_demand > 0
    # pheromone = phero_mats[v_type, position, interesting_idx]    
    # closeness = 1 / distances[position, interesting_idx]    
    # weights = pheromone * closeness
    weights = cost_mat[v_type, position, interesting_idx]
    chosen = helpers.pick_weighted_index(weights)

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
            # when j==0: adds way (last_node -- depot)
            tour_length += distances[tours[i][j-1], tours[i][j]]

        cost += fuel_consumption * tour_length
    return(cost)

# add pheromone to all edges contained in a path
# amount of pheromone inversly proportional to the cost of that path
# parameter Q >= 1  can increase amout of pheromone
def add_pheromone(tours, cost, Q, v_idx):
    global phero_mats
    additional_pheromone = Q/cost
    # go through all tours
    for i in range(len(tours)):
        v_type = idx_2_type(v_idx[i])
        # go through all nodes in the tour and edd pheromone to the edges on the way
        for j in range(len(tours[i])):
            phero_mats[v_type, tours[i][j-1], tours[i][j]] += additional_pheromone


# evaporation and intensification
def pheromone_changes(tours, vehicle_idx, evaporation_rate=0.01, Q=100):
    # evaporation:
    global phero_mats
    phero_mats = (1-evaporation_rate) * phero_mats

    # intensification:
    # calculate cost of path
    cost = path_cost(tours, vehicle_idx)
    # add pheromone to all edges included in path
    add_pheromone(tours, cost, Q, vehicle_idx)
    return cost


def collect_several_solutions(vehicle_idx=None, batch_size=100, keep_v=0, enforce_diverse_start=False):
    sol_list = []
    if enforce_diverse_start:
        # have a shuffeled list of indices of all customers
        # take the first ones in this list
        customers = np.arange(len(demand))
        np.random.shuffle(customers)
    for i in range(1, batch_size+1):
        if(vehicle_idx and i<len(vehicle_idx)):
            if(enforce_diverse_start):
                sol_list.append(solution_generation(visit_first=customers[i], vehicles=vehicle_idx[i]))
            else:
                sol_list.append(solution_generation(vehicles=vehicle_idx[i]))
        else: # not specifying the order of vehicles to use
            if(enforce_diverse_start):
                sol_list.append(solution_generation(visit_first=customers[i]))
            else:
                sol_list.append(solution_generation())
    cost_arr = np.zeros((len(sol_list,)))
    for i in range(len(sol_list)):
        cost = pheromone_changes(sol_list[i][0], sol_list[i][1])
        cost_arr[i] = cost
    # print('average and min over 100 runs: ', np.mean(cost_arr), np.min(cost_arr))
    print(min(cost_arr))
    idx_good_solutions = np.argsort(cost_arr)
    collect_vehicle_assignments = []
    
    # find the keep_v best vehicle assignments and return them, 
    # so that they can be reused in the next iteration
    for i in range(keep_v):
        # print(idx_good_solutions[i])
        collect_vehicle_assignments.append(sol_list[idx_good_solutions[i]][1])
    '''
        # to see which vehicles were used (gives a lot of print output)
        for_print = []
        for j in range(19):
            for_print.append(idx_2_type(collect_vehicle_assignments[i][j]))
        print(for_print)
    print(collect_vehicle_assignments[0][:10])
    '''
    return collect_vehicle_assignments, np.mean(cost_arr), np.min(cost_arr)
  

def do_iterations(iterations, batch_size=100, keep_v=0, enforce_diverse_start=0):
    if (batch_size > len(demand)):
        batch_size = len(demand) - 1
        print('reduced the batch size to', batch_size, ', the number of customers in this problem')
    if (keep_v > batch_size):
        keep_v = int(np.floor(0.5 * batch_size))
        print('reduced the number of vehicles to keep to', keep_v, ' half the batch size')
    enforce_until = 0
    if (enforce_diverse_start > 0):
        if (enforce_diverse_start > 1):
            print('ignoring this enforce_diverse_start value, expect something between 0 and 1')
        else:
            enforce_until = int(np.floor(enforce_diverse_start * iterations))
            print('enforce a diverse start until', enforce_until)
    value_history = np.zeros((iterations, 2))
    v_assign = None
    for i in range(iterations):
        if (i > enforce_until):
            enforce_diverse_start = False
        v_assign, meanV, minV = collect_several_solutions(v_assign, batch_size, keep_v, enforce_diverse_start)
        value_history[i, 0] = meanV
        value_history[i, 1] = minV

    # plt.figure()
    # plt.suptitle('Mean and min of the batches')
    # x = np.linspace(0, iterations*batch_size, num=iterations)
    # plt.plot(x, value_history[:,0])
    # plt.plot(x, value_history[:,1])
    # plt.axvline(enforce_until*batch_size)
    # plt.xlabel('single runs')
    # plt.ylabel('cost')
    # plt.show()
    '''
    # just to get some feeling for the values in the pheromone_mat
    f, axarr = plt.subplots(2,3, sharex=True, sharey=True)
    plt.suptitle('part of the pheromone matrices')
    cnt = 0
    axarr[0,0].imshow(phero_mats[0,:30,:30], interpolation='none', origin='bottom', cmap='jet')
    axarr[0,0].set_title('first vehicle')
    axarr[0,1].imshow(phero_mats[1,:30,:30], interpolation='none', origin='bottom', cmap='jet')
    axarr[0,1].set_title('second vehicle')
    axarr[1,0].imshow(phero_mats[2,:30,:30], interpolation='none', origin='bottom', cmap='jet')
    axarr[1,0].set_title('third vehicle')
    axarr[1,1].imshow(phero_mats[3,:30,:30], interpolation='none', origin='bottom', cmap='jet')
    axarr[1,1].set_title('fourth vehicle')
    axarr[0,2].imshow(distances[:30,:30], interpolation='none', origin='bottom', cmap='jet')
    axarr[0,2].set_title('distances')
    axarr[1,2].imshow(np.mean(phero_mats[:,:30,:30], axis=0), interpolation='none', origin='bottom', cmap='jet')
    axarr[1,2].set_title('mean all vehicles')
    plt.show()
    print('The values (mean, max, min) of the pheromone matrices are:')
    for j in range(4):
        print('matrix ', j)
        print(np.mean(phero_mats[j,:,:]))
        print(np.max(phero_mats[j,:,:]))
        print(np.min(phero_mats[j,:,:]))
     '''
