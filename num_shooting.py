import numpy as np
import matplotlib.pyplot as plt
import math
from solve_odes import solve_ode
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp


def pp_eqs(U0, t, args):
    """
    A function that returns the values of the given predator prey functions at (U0, t)
    :param X: values for x and y
    :param t: time
    :param args: list of a, b and d constants
    :return: solutions to predator prey equations
    """

    x, y = U0

    a = args[0]
    b = args[1]
    d = args[2]

    dxdt = x*(1-x) - a*x*y/(d+x)
    dydt = b*y*(1-(y/x))

    return [dxdt, dydt]


def hopf(U0, t, args):
    """
    A function that returns solutions to the hopf equations at (U0, t)
    :param U0: values for x and y
    :param t: time
    :param args: list of beta and sigma constants
    :return: solutions to hopf equations
    """

    beta = args[0]
    sigma = args[1]
    u1, u2 = U0
    du1dt = beta * u1 - u2 + sigma * u1 * (u1**2 + u2**2)
    du2dt = u1 + beta * u2 + sigma * u2 * (u1**2 + u2**2)

    return [du1dt, du2dt]


def pc(U0, T,  ode, *args):
    return ode(U0, T, args)[0]


def shooting(ode):

    def conditions(U0, *args):

        X0, T = U0[:-1], U0[-1]
        sols = solve_ode(X0, 0, T, ode, 'runge', 0.01, *args)
        sol = sols[-1, :]
        phase_cond = ode(X0, 0, *args)[0]
        period_cond = [X0[0] - sol[0], X0[1] - sol[1]]
        if len(sol) > 2:
            period_cond = []
            for num in range(len(sol)):
                i_period_cond = [X0[num] - sol[num]]
                period_cond = period_cond + i_period_cond

        return np.r_[phase_cond, period_cond]

    return conditions


def orbit(ode, U0, *args):
    sol = fsolve(shooting(ode), U0, *args)
    return sol


def shooting_plots(U0, ode, *args):

    sol = orbit(ode, U0, args)
    X0, T = sol[:-1], sol[-1]
    sol_mine = solve_ode(X0, 0, T, ode, 'runge', 0.01, *args)
    plt.plot(sol_mine[:, 0], sol_mine[:, 1])
    plt.show()


def shooting_one_cycle(U0, ode, *args):

    sol = orbit(ode, U0, args)
    X0, T = sol[:-1], sol[-1]
    sol_mine = solve_ode(X0, 0, T, ode, 'runge', 0.01, *args)
    time_cycle = math.ceil(float(T)/0.01) + 1
    t = np.linspace(0, T, time_cycle)
    plt.plot(t, sol_mine)
    plt.show()


# # initial guess predator-prey
# X0 = 0.2, 0.3
# T = 21

# initial guess hopf
method = 'runge'
ode = hopf
args = [1, -1]
U0 = 1.5, 1.5, 5
X0 = 1.5, 1.5
T = 5
deltat_max = 0.01

# print(orbit(ode, U0, args))

# X0 = 1.6, 1.2

# # initial guess hopf 3 odes
# X0 = 1, 1, 1
# T = 8

# shooting_plots(U0, ode, args)
# shooting_one_cycle(U0, ode, args)
# test_hopf_solutions(X0, T, ode, method, deltat_max, *args)
