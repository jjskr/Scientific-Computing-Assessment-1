import numpy as np
import matplotlib.pyplot as plt
import math
from solve_odes import solve_ode
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from test_for_shooting import hopf
from test_for_shooting import hopf_3
from test_for_shooting import true_sol


def pp_eqs(X, t, args):

    x, y = X

    # a = 1
    # b = 0.16
    # d = 0.1

    a = args[0]
    b = args[1]
    d = args[2]

    dxdt = x*(1-x) - a*x*y/(d+x)
    dydt = b*y*(1-(y/x))

    return [dxdt, dydt]


def shooting(X0, T, ode, method, *args):

    def conditions(U0):

        X0, T = U0[:-1], U0[-1]
        sols = solve_ode(X0, 0, T, ode, method, 0.01, *args)
        sol = sols[-1, :]
        phase_cond = ode(X0, 0, *args)[0]
        period_cond = [X0[0] - sol[0], X0[1] - sol[1]]
        if len(sol) > 2:
            period_cond = []
            for num in range(len(sol)):
                i_period_cond = [X0[num] - sol[num]]
                period_cond = period_cond + i_period_cond

        # print(X0, sol, T)

        return np.r_[phase_cond, period_cond]

    sol = fsolve(conditions, np.r_[X0, T])

    return sol


def shooting_plots(X0, T, ode, method, *args):

    sol = shooting(X0, T, ode, method, *args)
    X0, T = sol[:-1], sol[-1]
    sol_mine = solve_ode(X0, 0, T, ode, method, 0.01, *args)
    plt.plot(sol_mine[:, 0], sol_mine[:, 1])
    plt.show()


def shooting_one_cycle(X0, T, ode, method, *args):

    sol = shooting(X0, T, ode, method, *args)
    X0, T = sol[:-1], sol[-1]
    sol_mine = solve_ode(X0, 0, T, ode, method, 0.01, *args)
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
X0 = 1.5, 1.5
T = 4
deltat_max = 0.01

# X0 = 1.6, 1.2

# # initial guess hopf 3 odes
# X0 = 1, 1, 1
# T = 8

# shooting_plots(X0, T, hopf, method, args)
# shooting_one_cycle(X0, T, hopf, method, args)

