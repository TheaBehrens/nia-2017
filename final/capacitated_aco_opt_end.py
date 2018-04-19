import csv
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import helpers
import two_opt
import time

# several functions have to know the distance and the pheromone of each path
distances = None
closeness = None
phero_mats = None
capacity = None
demand = None
t_cost = None
types = None
each_vehicle_once = None


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
    global distances, phero_mats, capacity, demand, t_cost, each_vehicle_once, types, closeness
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
    closeness = 1.0 / (distances + 1e-20)
    closeness = closeness
    t_cost = np.asarray(t_cost, dtype=int).squeeze()     # (33,)
    types, each_vehicle_once = np.unique(capacity, return_index=True)

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
def solution_generation(visit_first=None, vehicles=None, alpha=1, beta=1, gamma=1, all_vehicles=False, iteration=0):

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
        if(all_vehicles):
            important_vehicles = each_vehicle_once
            np.random.shuffle(important_vehicles)
            vehicles = np.concatenate([important_vehicles, vehicles])

    # preference_mat is the basis for an ant's decision which customer to visit next, based on
    # - pheromone (different pheromone trails for different vehicle types)
    # - the distance (closer is prefered)
    # theoretically this would be the place to take into account the weights alpha, beta
    # (and gamma, if one would also include the open demand of the customer and the left stock of the ant)
    # but at the moment alpha and beta are to be taken as 1, both contribute equally
    preference_mat = np.zeros_like(phero_mats)
    for i in range(phero_mats.shape[0]):
        temp = np.power(phero_mats[i, :, :], 4)
        closeWeight = closeness
        denominator = np.sum(temp * closeWeight)
        preference_mat[i, :, :] = temp * closeWeight / denominator

    # continue until all customers are served
    i = 0
    while(sum(open_demand) != 0):
        # take the first vehicle and let it serve customers until empty
        left_stock = capacity[vehicles[i]]
        route = [0] # start at depot
        v_type = idx_2_type(vehicles[i])
        while(left_stock > 0):
            # the current position of the ant is written at the last position of route
            next_customer = choose_customer(route[-1], open_demand, left_stock, v_type, preference_mat)
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
    for i in range(len(types)):
        if capacity[idx] == types[i]:
            return i

# of all remaining nodes, chose one based on
#   - the pheromone,
#   - the distance,
#   - demand/stock --> took that part out again, because it only slowed down and did not improve the solution
def choose_customer(position, open_demand, stock, v_type, preference_mat):
    interesting_idx = open_demand > 0
    weights = preference_mat[v_type, position, interesting_idx]
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


def collect_several_solutions(vehicle_idx=None,
                              batch_size=100,
                              keep_v=0,
                              enforce_diverse_start=False,
                              all_vehicles=False,
                              iteration=0):
    sol_list = []
    if enforce_diverse_start:
        # have a shuffeled list of indices of all customers
        # take the first ones in this list
        customers = np.arange(len(demand))
        np.random.shuffle(customers)
    for i in range(1, batch_size+1):
        if(vehicle_idx and i<len(vehicle_idx)):
            if(enforce_diverse_start and all_vehicles):
                sol_list.append(solution_generation(all_vehicles=True, iteration=iteration))
            if(enforce_diverse_start and not(all_vehicles)):
                sol_list.append(solution_generation(visit_first=customers[i],
                                                    vehicles=vehicle_idx[i],
                                                    iteration=iteration))

            else:
                sol_list.append(solution_generation(vehicles=vehicle_idx[i], iteration=iteration))
        else: # not specifying the order of vehicles to use
            if(enforce_diverse_start and all_vehicles):
                sol_list.append(solution_generation(all_vehicles=True, iteration=iteration))
            if(enforce_diverse_start and not(all_vehicles)):
                sol_list.append(solution_generation(visit_first=customers[i], iteration=iteration))
            else:
                sol_list.append(solution_generation(iteration=iteration))
    cost_arr = np.zeros((len(sol_list,)))
    for i in range(len(sol_list)):
        cost = pheromone_changes(sol_list[i][0], sol_list[i][1])
        cost_arr[i] = cost
    # print('average and min over 100 runs: ', np.mean(cost_arr), np.min(cost_arr))
#    print(min(cost_arr))
    idx_good_solutions = np.argsort(cost_arr)
    collect_vehicle_assignments = []

    # find the keep_v best vehicle assignments and return them,
    # so that they can be reused in the next iteration
    for i in range(keep_v):
        # print(idx_good_solutions[i])
        collect_vehicle_assignments.append(sol_list[idx_good_solutions[i]][1])
    best_solution = sol_list[idx_good_solutions[0]][0]
    # best_idx = idx_good_solutions[0]
    return collect_vehicle_assignments, np.mean(cost_arr), np.min(cost_arr), best_solution#, best_idx

def vehicle_color(v_index):
    v_type = idx_2_type(v_index)
    if(v_type == 0):
        return 'r'
    if(v_type == 1):
        return 'y'
    if(v_type == 2):
        return 'b'
    if(v_type == 3):
        return 'c'
    return 'k'




def do_iterations(iterations, batch_size=100, keep_v=0, enforce_diverse_start=0, all_vehicles=False, initial_pheromone = 0.0001, vehicle_proportion = 0):
    div_start = enforce_diverse_start
    if (batch_size > len(demand)):
        batch_size = len(demand) - 1
   #     print('reduced the batch size to', batch_size, ', the number of customers in this problem')
    if (keep_v > batch_size):
        keep_v = int(np.floor(0.5 * batch_size))
  #      print('reduced the number of vehicles to keep to', keep_v, ' half the batch size')
    enforce_until = 0
    if (enforce_diverse_start > 0):
#        if (enforce_diverse_start > 1):
 #           print('ignoring this enforce_diverse_start value, expect something between 0 and 1')
  #      else:
            enforce_until = int(np.floor(enforce_diverse_start * iterations))
#            print('enforce a diverse start until', enforce_until)
    value_history = np.zeros((iterations, 2))
    v_assign = None
    best_sol = None
    first_sol = None
    alltimeMinV = np.Inf # correct?
    for i in range(iterations):
        if (i > enforce_until):
            enforce_diverse_start = False
        v_assign, meanV, minV, current_best_sol = collect_several_solutions(v_assign, batch_size, keep_v+1, enforce_diverse_start, all_vehicles, i)
        if (i==1):
            first_sol = (current_best_sol, v_assign[0])
        if(minV <= alltimeMinV):
            alltimeMinV = minV
            best_sol = (current_best_sol, v_assign[0])
        value_history[i, 0] = meanV
        value_history[i, 1] = minV


    #############################################
    # adding 2 opt here to the best solution found
    cost_before = alltimeMinV # path_cost(solutions, vehicles)

    max_depth = 30

    tic = time.time()
    opt_solutions = [two_opt.optimize(s + [0], distances, max_depth=max_depth)[0] for s in best_sol[0]]
    solutions = [s[:-1] for s in opt_solutions]
    elapsed = time.time() - tic

    cost_after = path_cost(solutions, best_sol[1])
    gain = cost_before - cost_after
#    print("cost_before: ", cost_before, " cost after: ", cost_after, "gain: ", gain, "time needed:", elapsed)

    ##################################################

    # # visualize the found path:
    # # using singular value decomposition to find a way to plot the cities in 2d
    U, s, eigenVecs =  np.linalg.svd(distances, full_matrices=False)
    dims = 2
    projected = np.dot(distances, np.transpose(eigenVecs[:dims,:]))

    # # ideas to improve:
    # # - could base the location of the cities on another PCA or TSNE (?) algorithm
    # # - plot not only the one best solution, but several good ones for comparison
    fig, axes = plt.subplots(1,2, sharex=True, sharey=True)
    colors = 'rgbycmkrgbycmkrgbycmk'
    bs = best_sol[0]
    for i in range(len(bs)):
        for j in range(len(bs[i])):
            c = vehicle_color(best_sol[1][i])
            axes[0].plot([projected[bs[i][j-1],0], projected[bs[i][j], 0]],
                      [projected[bs[i][j-1],1], projected[bs[i][j], 1]], color=c)

    for i in range(len(solutions)):
        for j in range(len(solutions[i])):
            c = vehicle_color(best_sol[1][i])
            axes[1].plot([projected[solutions[i][j-1],0], projected[solutions[i][j], 0]],
                      [projected[solutions[i][j-1],1], projected[solutions[i][j], 1]], color=c)

    # # adding the customers (size dependent on demand):
    axes[0].scatter(projected[:,0], projected[:,1], s=(demand**2)/10)
    axes[1].scatter(projected[:,0], projected[:,1], s=(demand**2)/10)
    # # adding the depot
    axes[0].scatter(projected[0,0], projected[0,1], c='k', marker='x', s=100, linewidths=3)
    axes[1].scatter(projected[0,0], projected[0,1], c='k', marker='x', s=100, linewidths=3)
    # # adding titles
    title = "Best solution, " + str(alltimeMinV)
    axes[0].set_title(title)
    title = "Swapped, " + str(cost_after)
    axes[1].set_title(title)
    # turning the ticks off
    axes[0].tick_params(axis='both', which='both', bottom='off', top='off',
                       left='off', right='off', labelleft='off', labelbottom='off')
    axes[1].tick_params(axis='both', which='both', bottom='off', top='off',
                       left='off', right='off', labelleft='off', labelbottom='off')
    #plt.show()
    plt.savefig('Best_solutions_1_'+ str(iterations)+ '_'+str(initial_pheromone)+'_'+ str(batch_size)+'_'+ str(keep_v)+'_'+ str(div_start) + '_' + str(all_vehicles)+'.png')
    plt.close()


    fig, axes = plt.subplots(1,2, sharex=True, sharey=True)
    bs = solutions # take the swapped solution here
    for i in range(len(bs)):
        for j in range(len(bs[i])):
            c = vehicle_color(best_sol[1][i])
            axes[0].plot([projected[bs[i][j-1],0], projected[bs[i][j], 0]],
                      [projected[bs[i][j-1],1], projected[bs[i][j], 1]], color=c)
            axes[1].plot([projected[bs[i][j-1],0], projected[bs[i][j], 0]],
                      [projected[bs[i][j-1],1], projected[bs[i][j], 1]], color=colors[i])
    # adding the customers (size dependent on demand)
    axes[0].scatter(projected[:,0], projected[:,1], s=(demand**2)/10)
    # axes[1].scatter(projected[:,0], projected[:,1], s=(demand**2)/10)
    # adding the depot
    axes[0].scatter(projected[0,0], projected[0,1], c='k', marker='x', s=100, linewidths=3)
    axes[1].scatter(projected[0,0], projected[0,1], c='k', marker='x', s=100, linewidths=3)
    # turning the ticks off
    axes[0].tick_params(axis='both', which='both', bottom='off', top='off',
                       left='off', right='off', labelleft='off', labelbottom='off')
    axes[1].tick_params(axis='both', which='both', bottom='off', top='off',
                       left='off', right='off', labelleft='off', labelbottom='off')
    # adding titles
    axes[0].set_title("colors according to vehicle type")
    axes[1].set_title("different color for each vehicle")
    plt.suptitle("Best solution, two visualizations")
    #plt.show()
    plt.savefig('Best_solutions_2_'+ str(iterations)+ '_'+str(initial_pheromone)+'_'+ str(batch_size)+'_'+ str(keep_v)+'_'+ str(div_start) + '_' + str(all_vehicles)+'.png')
    plt.close()




    plt.figure()
    plt.suptitle('Mean and min of the batches')
    x = np.linspace(0, iterations*batch_size, num=iterations)
    plt.plot(x, value_history[:,0])
    plt.plot(x, value_history[:,1])
    plt.axvline(enforce_until*batch_size)
    plt.axhline(alltimeMinV)
    plt.axhline(cost_after)
    plt.ylim([50000, 250000])
    plt.xlabel('single runs')
    plt.ylabel('cost')
    #plt.show()

    plt.savefig('Mean_and_min_'+ str(iterations)+ '_'+str(initial_pheromone)+'_'+ str(batch_size)+'_'+ str(keep_v)+'_'+ str(div_start) + '_' + str(all_vehicles)+ '.png')
    plt.close()
    return cost_after
