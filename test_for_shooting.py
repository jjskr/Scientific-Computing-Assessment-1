import numpy as np


def hopf(U0, t, args):

    # beta = 1
    # sigma = -1

    beta = args[0]
    sigma = args[1]

    u1, u2 = U0
    du1dt = beta * u1 - u2 + sigma * u1 * (u1**2 + u2**2)
    du2dt = u1 + beta * u2 + sigma * u2 * (u1**2 + u2**2)

    return [du1dt, du2dt]


def true_sol(t, args):
    # beta = 1
    # theta = 0
    beta = args[0]
    theta = args[1]

    u1 = np.sqrt(beta) * np.cos(t + theta)
    u2 = np.sqrt(beta) * np.sin(t + theta)

    return [u1, u2]


def hopf_3(U0, t, args):

    # beta = 1
    # sigma = -1
    beta = args[0]
    sigma = args[1]

    u1, u2, u3 = U0

    du1dt = beta * u1 - u2 + sigma * u1 * (u1**2 + u2**2)
    du2dt = u1 + beta * u2 + sigma * u2 * (u1**2 + u2**2)
    du3dt = -u3

    return [du1dt, du2dt, du3dt]
