import numpy as np
import matplotlib.pyplot as plt



# points to fit a polynomial to:
x = np.linspace(0,10,300)
points = np.cos(x) + np.random.normal(0,0.1, 300)

bounds = np.array([[-2,2],[-2,2],[-2,2],[-1,1],[-1,1],[-1,1],[-1,1]])

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
        # plt.plot(x, curve)
        # plt.pause(0.5)
    return curve
'''
plt.ion()
plt.figure()
plt.plot(x, np.sin(x))
plt.scatter(x, points)
plt.pause(0.1)
plt.ylim(-2,2)
'''

# best fitness for lowes mean-square error:
def fitness(vector):
    y_pred = curve(vector)
    error = np.sqrt(sum((points - y_pred) ** 2) / len(points))
    return error

#fitness(np.array([1,0.5,-1,0.5,0.05]))
#fitness(np.array([ 0.99677643,  0.47572443, -1.39088333,  0.50950016, -0.06498931, 0.00273167]))
plt.pause(2)
