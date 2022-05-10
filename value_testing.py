import numpy as np
from solve_odes import solve_ode
from num_shooting import shooting
from num_shooting import orbit
from num_continuation import continuation
import math
from scipy.optimize import fsolve


def test_shooting_solutions(U0, ode, f_true_sol, args):

    """
    Function to test for correct solve_ode values for 2D and 3D ODEs
    :param U0: initial conditions
    :param ode: ODE(s) to solve for
    :param f_true_sol: true solution
    :param args: required arguments
    :return: test_status, 0 if passed, 1 if failed
    """

    test_status = 0

    orb_sol = orbit(ode, U0, pc_stable_0, args)

    X0, T = orb_sol[:-1], orb_sol[-1]

    # checks correct period is found
    if T - 2*math.pi > 10**-8:
        test_status = 1

    x_vals = solve_ode(X0, 0, T, ode, 'runge', 0.01, args)

    # checks start is equal to finish
    for i in range(len(x_vals[0])):
        errorr = x_vals[0][i] - x_vals[-1][i]
        if errorr > 10**-8:
            test_status = 1

    t_array = []
    t_array = t_array + [0]
    current_t = 0
    while T - current_t > 0.01:
        current_t += 0.01
        t_array = t_array + [current_t]
    if current_t != T:
        t_array = t_array + [T]

    # checking solve_ode produces correct values
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


if __name__ == '__main__':

    def cubic_eq(x, args):

        c = args
        cubic_sol = x ** 3 - x + c

        return cubic_sol


    def hopfn(U0, t, args):

        beta = args

        u1, u2 = U0

        du1dt = beta * u1 - u2 - u1 * (u1 ** 2 + u2 ** 2)
        du2dt = u1 + beta * u2 - u2 * (u1 ** 2 + u2 ** 2)

        return [du1dt, du2dt]


    def pc_stable_0(U0, T, ode, args):
        return ode(U0, 0, args)[0]


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
        du1dt = beta * u1 - u2 + sigma * u1 * (u1 ** 2 + u2 ** 2)
        du2dt = u1 + beta * u2 + sigma * u2 * (u1 ** 2 + u2 ** 2)

        return [du1dt, du2dt]


    def true_sol(t, args):
        """
        Function returning true hopf solution at t
        :param t: time value
        :param args: beta and theta values
        :return: hopf solution at t
        """

        beta = args[0]
        theta = args[1]

        u1 = np.sqrt(beta) * np.cos(t + theta)
        u2 = np.sqrt(beta) * np.sin(t + theta)

        return [u1, u2]


    def true_sol_3(t, args):
        """
        Function returning true solution for 3 dimensional hopf equation at t
        :param t: time value
        :param args: beta and theta values
        :return: true solution for 3 dimensional hopf equation at t
        """

        beta = args[0]
        theta = args[1]

        u1 = np.sqrt(beta) * np.cos(t + theta)
        u2 = np.sqrt(beta) * np.sin(t + theta)
        u3 = 0

        return [u1, u2, u3]


    def hopf_3(U0, t, args):
        """
        Function for 3 dimensional hopf equatiom
        :param U0: initial conditions
        :param t: time value
        :param args: beta and sigma values
        :return: values of dudt for all u values
        """
        beta = args[0]
        sigma = args[1]

        u1, u2, u3 = U0

        du1dt = beta * u1 - u2 + sigma * u1 * (u1 ** 2 + u2 ** 2)
        du2dt = u1 + beta * u2 + sigma * u2 * (u1 ** 2 + u2 ** 2)
        du3dt = -u3

        return [du1dt, du2dt, du3dt]

    # functions called to test the results obtained by solve_ode function and orbit to ensure correct values are
    # found. tests that time period found is approximately equal to 2*pi

    method = 'runge'
    ode = hopf
    args = [1, -1]
    U0 = 1.5, 1.5, 5
    X0 = 1.5, 1.5
    T = 5
    deltat_max = 0.01

    if test_shooting_solutions(U0, ode, true_sol, args) == 0:
        print('value test passed for ordinary hopf')
    else:
        print('value test failed for ordinary hopf')

    ode = hopf_3
    U0 = 1.5, 1.5, 1.5, 6
    X0 = 1.5, 1.5, 1.5
    T = 6

    if test_shooting_solutions(U0, ode, true_sol_3, args) == 0:
        print('value test passed for 3 dimensional hopf')
    else:
        print('value test failed for 3 dimensional hopf')

    # testing natural parameter continuation errors for hopf

    U0 = 1.4, 0, 6.3
    pmin = 2
    pmax = 0
    pstep = 20

    solutions, parameters = continuation('natural', hopfn, U0, pmin, pmax, pstep, pc_stable_0, shooting, fsolve)
    t_list = []
    exact_sols = []
    j = 0
    for i in solutions:
        t = i[-1]
        args = [parameters[j], 0]
        exact_sol = true_sol(t, args)
        exact_sols = exact_sols + [exact_sol]
        j += 1

    error = 0
    for i in range(len(exact_sols)):
        error1 = abs(solutions[i][0] - exact_sols[i][0])
        error2 = abs(solutions[i][1] - exact_sols[i][1])
        if error1 > 10**-5:
            error = 1
        elif error2 > 10**-5:
            error = 1

    if error == 0:
        print('value test passed for continuation')
    else:
        print('value test failed for continuation')
