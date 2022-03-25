import numpy as np
import matplotlib.pyplot as plt
import math
from solve_odes import solve_ode
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp


def pp_eqs(X, t):

    x, y = X

    a = 1
    b = 0.16
    d = 0.1

    dxdt = x*(1-x) - a*x*y/(d+x)
    dydt = b*y*(1-(y/x))

    return [dxdt, dydt]


def conditions(U0):

    X0, T = U0[:-1], U0[-1]
    sols = solve_ode(X0, 0, T, pp_eqs)
    sol = sols[-1, :]
    phase_cond = pp_eqs(X0, 0)[0]
    period_cond = [X0[0] - sol[0], X0[1] - sol[1]]
    print(X0, sol, T)

    return np.r_[phase_cond, period_cond]


def shooting(X0, T):

    sol = fsolve(conditions, np.r_[X0, T])

    return sol


def shooting_plots(X0, T):

    sol = shooting(X0, T)
    X0, T = sol[:-1], sol[-1]
    sol_mine = solve_ode(X0, 0, T, pp_eqs)
    plt.plot(sol_mine[:, 0], sol_mine[:, 1])
    plt.show()


def shooting_one_cycle(X0, T):

    sol = shooting(X0, T)
    X0, T = sol[:-1], sol[-1]
    sol_mine = solve_ode(X0, 0, T, pp_eqs)
    plt.plot(sol_mine)
    plt.show()


# initial guess
X0 = 0.2, 0.3
T = 21

shooting_plots(X0, T)
