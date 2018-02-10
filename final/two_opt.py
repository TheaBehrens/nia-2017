import helpers
import numpy as np
from numba import jit

def optimize(path, distance_matrix, max_depth=None):
    cost = helpers.compute_path_cost(path, distance_matrix)
    opt_path, opt_cost =  two_opt_rec(path, cost, distance_matrix, 0, max_depth)
    #print("2-opt done, improment: ", cost - opt_cost)
    return opt_path, opt_cost

def optimize_step(path, distance_matrix):
    best_cost = helpers.compute_path_cost(path, distance_matrix)
    best_path = path
    for i in range(1, len(path)-1):
        for k in range(i+1, len(path)-1):
            candidate = two_opt_swap(path, i, k)
            cost = helpers.compute_path_cost(candidate, distance_matrix)
            if cost < best_cost:
                best_cost = cost
                best_path = candidate
    return best_path, best_cost

# private

# def two_opt_swap(path, i, k):
#     return path[0:i] + list(reversed(path[i:k+1])) + path[k+1:]

def two_opt_swap(path, i, k):
    copy = np.copy(path)
    copy[i:k+1] = path[k:i-1:-1]
    return copy

def two_opt_rec(path, last_cost, distance_matrix, depth, max_depth):
    if max_depth and depth >= max_depth:
        return path, last_cost
    opt_path, opt_cost = optimize_step(path, distance_matrix)
    if opt_cost < last_cost:
        #print("improved by: ", last_cost - opt_cost)
        return two_opt_rec(opt_path, opt_cost, distance_matrix, depth + 1, max_depth)
    else:
        return path, last_cost