from scipy.optimize import fsolve
from num_shooting import shooting
import matplotlib.pyplot as plt
# from num_shooting import hopf
import numpy as np
from pde_solving import solve_pde
from math import pi


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


def cubic_eq(x, args):

    c = args
    cubic_sol = x**3 - x + c

    return cubic_sol


def hopfn(U0, t, args):

    beta = args

    u1, u2 = U0

    du1dt = beta * u1 - u2 - u1 * (u1**2 + u2**2)
    du2dt = u1 + beta * u2 - u2 * (u1**2 + u2**2)

    return [du1dt, du2dt]


def hopfm(U0, t, args):

    beta = args

    u1, u2 = U0

    du1dt = beta * u1 - u2 + u1 * (u1**2 + u2**2) - u1 * (u1**2 + u2**2) ** 2
    du2dt = u1 + beta * u2 + u2 * (u1**2 + u2**2) - u2 * (u1**2 + u2**2) ** 2

    return [du1dt, du2dt]


def nat_continuation(ode, U0, par_min, par_max, par_split, pc, discretisation, solver=fsolve):
    """
    Performs natural parameter continuation on chosen set of ODEs returning parameter list and corresponding solution
    for each parameter value

    :param ode: System of ODEs to solve
    :param U0: Initial conditions
    :param par_min: Starting parameter value
    :param par_max: End parameter value
    :param par_split: Number of intervals between parameter range
    :param discretisation: Discretisation to use
    :param solver: Type of solver to use. (only implemented fsolve and cont_pde for pdes)
    :return: List of parameters and solutions at each parameter value
    """

    params = np.linspace(par_min, par_max, par_split)

    solut_list = []

    if solver == 'cont_pde':
        rou = 10
        solver = cont_pde
    else:
        rou = 5

    for par in params:
        if pc:
            args = (pc, par)
        else:
            args = par
        U0 = solver(discretisation(ode), U0, args)
        U0 = np.round(U0, rou)
        solut_list = solut_list + [U0]
    return params, solut_list


def psuedo_continuation(ode, U0, par_min, par_max, par_split, discretisation, solver=fsolve):
    """
    Performs pseudo-arclength continuation on chosen set of ODEs returning parameter list and
    corresponding solution for each parameter value

    :param ode: System of ODEs to solve
    :param U0: Initial conditions
    :param par_min: Starting parameter value
    :param par_max: End parameter value
    :param par_split: Number of intervals between parameter range
    :param discretisation: Discretisation to use
    :param solver: Type of solver to use. (only implemented fsolve and cont_pde for pdes)
    :return: List of parameters and solutions at each parameter value
    """

    if discretisation==shooting:
        R0 = list(U0)
        R0 = R0[:-1]

    def p_upd(pred):

        par = pred

        return par

    params = np.linspace(par_min, par_max, par_split)

    diff = params[1] - params[0]
    par0 = params[0]
    par1 = par0 + diff

    val0 = fsolve(discretisation(ode), U0, par0)
    val1 = fsolve(discretisation(ode), val0, par1)

    sol_lis = [val0[0], val1[0]]
    param_l = [par0, par1]

    i = 0

    par_add = par_min

    while par_add < par_max:

        # generate sec
        delta_x = sol_lis[i+1] - sol_lis[i]
        delta_p = param_l[i+1] - param_l[i]

        pred_x = sol_lis[i+1] + delta_x
        pred_p = param_l[i+1] + delta_p

        pred_ar = [pred_x, pred_p]

        pred_ar = np.array(pred_ar)

        sol = solver(lambda cur_s: np.append(discretisation(ode)(cur_s[:-1], p_upd(cur_s[-1])),
                                                    np.dot(cur_s[:-1] - pred_x, delta_x) + np.dot(cur_s[-1] - pred_p,
                                                                                                  delta_p)), pred_ar)

        sol_add = sol[:-1][0]
        par_add = sol[-1]
        sol_lis = sol_lis + [sol_add]
        param_l = param_l + [par_add]

        i += 1

    return sol_lis, param_l


if __name__ == '__main__':

    # # cubic initial conditions
    # U0 = 1
    # pmin = -2
    # pmax = 2
    # pstep = 100
    #
    # par_list, solutions = nat_continuation(cubic_eq, U0, pmin, pmax, pstep, None, lambda x: x)
    # plt.plot(par_list, solutions, label='natural parameter')
    # #
    # U0 = 1
    # pmin = -2
    # pmax = 2
    # pstep = 100
    #
    # sol_l, p_l = psuedo_continuation(cubic_eq, U0, pmin, pmax, pstep, lambda x: x)
    # plt.plot(p_l, sol_l, label='pseudo-arclength')
    # plt.legend()
    # plt.show()

    # hopf continuation - natural works but could not get pseudo arclength to work

    U0 = 1.4, 0, 6.3
    pmin = 2
    pmax = 0
    pstep = 50

    # par_list, solutions = nat_continuation(hopfn, U0, pmin, pmax, pstep, pc_stable_0, shooting)
    # plt.plot(par_list, solutions)
    # plt.show()

    pmin = 2
    pmax = -1
    pstep = 34

    # par_list, solutions = nat_continuation(hopfm, U0, pmin, pmax, pstep, pc_stable_0, shooting)
    # plt.plot(par_list, solutions)
    # plt.show()

    # U0 = 1.4, 0, 6.3
    # params = [2, -1]
    # pmin = -1
    # pmax = 2
    # pstep = 0.1
    #
    # par_list, solutions = nat_continuation_h(hopf, U0, params, pmin, pmax, pstep, 0, shooting)

    U0 = 1.4, 0, 6.3
    pmin = -1
    pmax = 2
    pstep = 100

    # sol_l, p_l = psuedo_continuation(hopfn, U0, pmin, pmax, pstep, shooting)
    # plt.plot(p_l, sol_l)
    # plt.show()

    # Attempting pde continuation

    kappa = 1.0  # diffusion constant
    L = 1.0  # length of spatial domain
    T = 0.5  # total time to solve for

    mx = 30  # number of gridpoints in space
    mt = 1000  # number of gridpoints in time


    def u_initial(x):
        # initial temperature distribution
        y = np.sin(pi * x / L)
        return y


    def p(t):
        return 3


    def q(t):
        return 4


    def pdef(U0, arg):
        print(U0)

        args = [arg, 1, 0.5]
        u_j = solve_pde(U0, mx, mt, 'CN', 'dirichlet', p, q, args)

        return u_j


    def cont_pde(f, U0, args):
        return f(U0, args)


    param, sols = nat_continuation(pdef, np.zeros(mx+1), 0.5, 2, 11, None, lambda x: x, 'cont_pde')
    t = np.linspace(0, T, mx + 1)  # mesh points in time

    j = 0

    for i in sols:
        ka = np.round(param[j], 4)
        ka = str(ka)
        plt.plot(t, i, label='T = ' + ka)
        plt.legend()
        j += 1
    plt.show()
    # solve_pde(mx, mt, 'CN', 'dirichlet', p, q, 2)
