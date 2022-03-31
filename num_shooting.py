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


def shooting(X0, T, ode, method, deltat_max, *args):

    def conditions(U0):

        X0, T = U0[:-1], U0[-1]
        sols = solve_ode(X0, 0, T, ode, method, deltat_max, *args)
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


def shooting_plots(X0, T, ode, method, deltat_max, *args):

    sol = shooting(X0, T, ode, method, deltat_max, *args)
    X0, T = sol[:-1], sol[-1]
    sol_mine = solve_ode(X0, 0, T, ode, method, deltat_max, *args)
    plt.plot(sol_mine[:, 0], sol_mine[:, 1])
    plt.show()


def shooting_one_cycle(X0, T, ode, method, deltat_max, *args):

    sol = shooting(X0, T, ode, method, deltat_max, *args)
    X0, T = sol[:-1], sol[-1]
    sol_mine = solve_ode(X0, 0, T, ode, method, deltat_max, *args)
    time_cycle = math.ceil(float(T)/0.01) + 1
    t = np.linspace(0, T, time_cycle)
    plt.plot(t, sol_mine)
    plt.show()


def test_hopf_solutions(X0, T, ode, method, deltat_max, *args):

    test_status = 0

    hopf_sol = shooting(X0, T, ode, method, deltat_max, args)

    X0, T = hopf_sol[:-1], hopf_sol[-1]
    # print(X0, T)

    x_vals = solve_ode(X0, 0, T, ode, method, deltat_max, args)

    # for i in x_vals:
    #     print(i[0])

    t_array = []
    t_array = t_array + [0]
    current_t = 0
    while T - current_t > deltat_max:
        current_t += deltat_max
        t_array = t_array + [current_t]
    if current_t != T:
        t_array = t_array + [T]

    actual = []
    for i in range(0, len(t_array)):
        t = t_array[i]
        sol = true_sol(t, [1, 0])
        actual = actual + [sol]

    for i in range(0, len(x_vals)):
        error1 = abs(actual[i][0] - x_vals[i][0])
        error2 = abs(actual[i][1] - x_vals[i][1])
        if error2 > 1*10**-6:
            test_status = 1

    return test_status


# # initial guess predator-prey
# X0 = 0.2, 0.3
# T = 21

# initial guess hopf
method = 'runge'
ode = hopf
args = [1, -1]
X0 = 1.5, 1.5
T = 5
deltat_max = 0.01

# X0 = 1.6, 1.2

# # initial guess hopf 3 odes
# X0 = 1, 1, 1
# T = 8

# shooting_plots(X0, T, hopf, method, deltat_max, args)
# shooting_one_cycle(X0, T, hopf, method, deltat_max, args)
# test_hopf_solutions(X0, T, ode, method, deltat_max, *args)
