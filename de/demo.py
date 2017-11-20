import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.animation
import numpy as np
import time

# algorithm parameters
POPULATION_SIZE = 30
MIN, MAX = -15, 15
NUM_PARAMS = 2
NUM_ITERATIONS = 150
CR = 0.1
F = 0.1

# parameters for ackley function
a = 10
b = 0.4
c = 0.5*np.pi
d = 2

def fitness(p):
    ssq = np.sum(np.power(p,2))
    scs = np.sum(np.cos(c*p))
    result = -a * np.exp(-b*ssq/d) - np.exp(scs/d) + a + np.exp(1)
    return result

def de_step(pop):
    new_pop = np.copy(pop)
    for i, x in enumerate(pop):
        u,v,w = pop[np.random.choice(len(pop), size=3, replace=False)]
        while u is x or u is x or u is x:
            u,v,w = pop[np.random.choice(len(pop), size=3, replace=False)]
        r = np.random.randint(0, NUM_PARAMS)
        p = np.random.rand(NUM_PARAMS) < CR
        p[r] = True
        z = u + F*(v-w)
        y = np.where(p, z, x)
        if fitness(y) < fitness(x):
            new_pop[i] = y
    return new_pop

domain = 15
samples = 51

X, Y = np.meshgrid(np.linspace(-domain, domain, samples), np.linspace(-domain, domain, samples))
ssq = np.power(X,2) + np.power(Y,2)
scs = np.cos(c*X) + np.cos(c*Y)
Z = -a * np.exp(-b*ssq/d) - np.exp(scs/d) + a + np.exp(1)

population = np.random.rand(POPULATION_SIZE, NUM_PARAMS) * (MAX-MIN) + MIN

fig = plt.figure()
fit = np.apply_along_axis(fitness, 1, population) 
ax = fig.add_subplot(111)
ax.contour(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0.3, edgecolor=(0,0,0,0.3))
scatter_plot = ax.scatter(population[:, 0], population[:, 1], fit)   
ax.set_xlim([-15,15]) 
ax.set_ylim([-15,15])

plt.ion()
plt.show()

for i in range(0, 50):
    population = de_step(population)
    fit = np.apply_along_axis(fitness, 1, population)
    scatter_plot.set_offsets(population)
    scatter_plot.set_array(fit)
    fig.canvas.draw()
    plt.pause(0.05)