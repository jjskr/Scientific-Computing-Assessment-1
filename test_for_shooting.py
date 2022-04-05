import numpy as np
from solve_odes import solve_ode
from num_shooting import shooting


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
        if error1 > 1*10**-6:
            test_status = 1

    return test_status
