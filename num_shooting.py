import numpy as np
import matplotlib.pyplot as plt
import math
from solve_odes import solve_ode
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from test_for_shooting import hopf


def pp_eqs(X, t):

    x, y = X

    a = 1
    b = 0.16
    d = 0.1

    dxdt = x*(1-x) - a*x*y/(d+x)
    dydt = b*y*(1-(y/x))

    return [dxdt, dydt]


def shooting(X0, T, ode):

    def conditions(U0):

        X0, T = U0[:-1], U0[-1]
        sols = solve_ode(X0, 0, T, ode, 'runge', 0.01)
        sol = sols[-1, :]
        phase_cond = ode(X0, 0)[0]
        period_cond = [X0[0] - sol[0], X0[1] - sol[1]]
        if len(sol) > 2:
            period_cond = []
            for num in range(sol):
                i_period_cond = [X0[num] - sol[num]]
                period_cond = period_cond + i_period_cond

        # print(X0, sol, T)

        return np.r_[phase_cond, period_cond]

    sol = fsolve(conditions, np.r_[X0, T])

    return sol


def shooting_plots(X0, T, ode):

    sol = shooting(X0, T, ode)
    X0, T = sol[:-1], sol[-1]
    sol_mine = solve_ode(X0, 0, T, ode, 'runge', 0.01)
    plt.plot(sol_mine[:, 0], sol_mine[:, 1])
    plt.show()


def shooting_one_cycle(X0, T, ode):

    sol = shooting(X0, T, ode)
    X0, T = sol[:-1], sol[-1]
    sol_mine = solve_ode(X0, 0, T, ode, 'runge', 0.01)
    time_cycle = math.ceil(float(T)/0.01) + 1
    t = np.linspace(0, T, time_cycle)
    plt.plot(t, sol_mine)
    plt.show()


# # initial guess predator-prey
# X0 = 0.2, 0.3
# T = 21

# initial guess hopf
X0 = 1.5, 1.5
T = 4

shooting_plots(X0, T, hopf)
shooting_one_cycle(X0, T, hopf)

