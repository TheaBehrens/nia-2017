import numpy as np
import matplotlib.pyplot as plt


nr_datapoints = 300

# points to fit a polynomial to:
x = np.linspace(0,10,nr_datapoints)
points = np.cos(x) + np.random.normal(0,0.1, nr_datapoints)

# the dimension of the bounds array determines
# the dimensionality of the search space
bounds = np.array([[-2,2],[-2,2],[-2,2],[-1,1],[-1,1],[-1,1],[-1,1]])

# # alternative problem:
# 
# # simple parabola
# nr_datapoints = 50
# x = np.linspace(-2,2,nr_datapoints)
# points = x**2 + np.random.normal(-1.5,0.9,nr_datapoints)
# points = x**2 + np.random.normal(-1.5,0.2,nr_datapoints)
# 
# 
# # too many dimensions, should cause some overfit:
# # especially for the parabola
# bounds = np.array([[-2,2],[-2,2],[-2,2],[-1,1],[-1,1],[-1,1],[-1,1],[-0.2,0.2],[-0.2,0.2],[-0.2,0.2],[-0.1,0.1],[-0.1,0.1],[-0.1,0.1],[-0.1,0.1]])

def get_bounds():
    return bounds

def get_x():
    return x

def get_points():
    return points

def curve(vector):
    # how many dimensions has the vector?
    dim = len(vector)
    curve = np.zeros(x.shape)
    for i in range(dim):
        curve += vector[i] * (x ** i)
    return curve

# best objective value for lowest mean-square error:
def objective_func(vector):
    y_pred = curve(vector)
    error = np.sqrt(sum((points - y_pred) ** 2) / len(points))
    return error

