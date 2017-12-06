import numpy as np
import matplotlib.pyplot as plt


def func_h(x0, x1, tau):
    if(min(tau) < 0 or max(tau) > 20):
        print('tau has to be in [0,20] but is ', tau)
        return 0
    return 1 / (1 + x0 * tau + (x1 * tau) * (x1 * tau))


def fitness(vector):
    x0 = vector[0]
    x1 = vector[1]
    tau = np.linspace(0, 20, 20)
    w = [1, 1, 1]
    value = 0
    gam1 = gamma1(x0, x1, tau)
    gam2 = gamma2(x0, x1, tau)
    gam3 = gamma3(x0, x1, tau)
    value += sum(gam1[gam1>0]*gam1[gam1>0])
    value += sum(gam2[gam2>0]*gam2[gam2>0])
    value += sum(gam3[gam3>0]*gam3[gam3>0])
    return value

def gamma1(x0, x1, tau):
    t = tau[tau < 10]
    return (func_h(x0, x1, t) - 1.04)
    # im buch ist es so rum:
    # return (1.04 - func_h(x0, x1, t))

def gamma2(x0, x1, tau):
    t = tau[tau >= 10]
    return (func_h(x0, x1, t) - 0.4)

def gamma3(x0, x1, tau):
    t = tau[tau <= 5]
    return (0.8 - func_h(x0, x1, t))

