import numpy as np


def hopf(U0, t):

    beta = 1
    sigma = -1
    u1, u2 = U0
    du1dt = beta * u1 - u2 + sigma * u1 * (u1**2 + u2**2)
    du2dt = u1 + beta * u2 + sigma * u2 * (u1**2 + u2**2)

    return [du1dt, du2dt]


def true_sol(t):

    beta = 1
    theta = 0
    u1 = np.sqrt(beta) * np.cos(t + theta)
    u2 = np.sqrt(beta) * np.sin(t + theta)

    return [u1, u2]

