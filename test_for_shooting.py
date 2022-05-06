import numpy as np
from solve_odes import solve_ode
from num_shooting import shooting
from num_shooting import orbit


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


def true_sol(t, args):

    beta = args[0]
    theta = args[1]

    u1 = np.sqrt(beta) * np.cos(t + theta)
    u2 = np.sqrt(beta) * np.sin(t + theta)

    return [u1, u2]


def true_sol_3(t, args):

    beta = args[0]
    theta = args[1]

    u1 = np.sqrt(beta) * np.cos(t + theta)
    u2 = np.sqrt(beta) * np.sin(t + theta)
    u3 = 0

    return [u1, u2, u3]


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


def test_shooting_solutions(U0, ode, f_true_sol, args):
    """
    Function to test for correct orbit and solve_ode values for 2D and 3D ODEs
    :param U0: initial conditions
    :param ode: ODE(s) to solve for
    :param deltat_max:
    :param f_true_sol:
    :param args:
    :return:
    """

    test_status = 0

    # hopf_sol = shooting(X0, T, ode, method, deltat_max, args)
    orb_sol = orbit(ode, U0, args)

    X0, T = orb_sol[:-1], orb_sol[-1]

    x_vals = solve_ode(X0, 0, T, ode, 'runge', 0.01, args)

    t_array = []
    t_array = t_array + [0]
    current_t = 0
    while T - current_t > 0.01:
        current_t += 0.01
        t_array = t_array + [current_t]
    if current_t != T:
        t_array = t_array + [T]

    actual = []
    for i in range(0, len(t_array)):
        t = t_array[i]
        sol = f_true_sol(t, [1, 0])
        actual = actual + [sol]

    if len(actual[0]) == 2:
        for i in range(0, len(x_vals)):
            error1 = abs(actual[i][0] - x_vals[i][0])
            error2 = abs(actual[i][1] - x_vals[i][1])
            if error2 > 1*10**-8:
                test_status = 1
            if error1 > 1*10**-8:
                test_status = 1

    elif len(actual[0]) == 3:
        for i in range(0, len(x_vals)):
            error1 = abs(actual[i][0] - x_vals[i][0])
            error2 = abs(actual[i][1] - x_vals[i][1])
            error3 = abs(actual[i][2] - x_vals[i][2])
            if error2 > 1*10**-8:
                test_status = 1
            if error1 > 1*10**-8:
                test_status = 1
            if error3 > 1*10**-8:
                test_status = 1

    return test_status


# method = 'runge'
# ode = hopf
args = [1, -1]
# U0 = 1.5, 1.5, 5
# X0 = 1.5, 1.5
# T = 5
#
# deltat_max = 0.01
#
# print(test_shooting_solutions(U0, ode, true_sol, args))
#
#
ode = hopf_3
U0 = 1.5, 1.5, 1.5, 6
X0 = 1.5, 1.5, 1.5
T = 6

print(test_shooting_solutions(U0, ode, true_sol_3, args))
