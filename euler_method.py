import numpy as np
import matplotlib.pyplot as plt


def f(x, t):
    return x


def euler_step(x, t):
    x = x + deltat_max*f(x, t)
    t = t + deltat_max
    print(x, round(t, 5))
    return x, t


def solve_to(x, t, t2):
    array_x = []
    array_t = []
    while t < t1:
        x, t = euler_step(x, t)
        array_x = array_x + [x]
        array_t = array_t + [round(t, 5)]
    plt.plot(array_t, array_x)
    plt.show()
    return array_x, array_t


# Initial Conditions
x0 = 1
x1 = 10
t0 = 0
t1 = 1
deltat_max = 0.2

solve_to(x0, t0, t1)

