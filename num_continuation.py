from scipy.optimize import fsolve
from num_shooting import shooting
import matplotlib.pyplot as plt
from num_shooting import hopf
import numpy as np
from pde_solving import solve_pde
from math import pi


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


def nat_continuation(f, U0, par_min, par_max, par_split, discretisation, solver=fsolve):
    """
    Performs natural parameter continuation on chosen set of ODEs returning parameter list and corresponding solution
    for each parameter value

    :param f: System of ODEs to solve
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
        rou = 10

    for par in params:
        U0 = solver(discretisation(f), U0, par)
        U0 = np.round(U0, rou)
        solut_list = solut_list + [U0]

    return params, solut_list


def psuedo_continuation(ode, U0, par_min, par_max, par_split, discretisation):

    params = np.linspace(par_min, par_max, par_split)

    diff = params[1] - params[0]
    par0 = params[0]
    par1 = par0 + diff

    params, solsnat = nat_continuation(ode, U0, par_min, par_max, par_split, discretisation)

    val0 = fsolve(discretisation(ode), U0, par0)
    val1 = fsolve(discretisation(ode), val0, par1)

    print(val1, solsnat[1])

    sol_lis = [val0[0], val1[0]]
    param_l = [par0, par1]

    i = 0

    while i < 100:
        print(sol_lis)
        # generate sec
        delta_x = sol_lis[i+1] - sol_lis[i]
        delta_p = param_l[i+1] - param_l[i]

        pred_x = sol_lis[i+1] + delta_x
        pred_p = param_l[i+1] + delta_p

        pred_ar = [pred_x, pred_p]

        psuedo_arc = np.dot(sol_lis[i+1] - pred_x, delta_x) + np.dot(param_l[i+1] - pred_p, delta_p)

        pred_sol = pred_x, pred_p

        n_par = pred_p

        i += 1

    # solution

    # solutions = solutions + [sol]
    # pars = pars + [par]
    #
    # val1 = sol
    # val0 = val1

    return val0, val1


if __name__ == '__main__':

    # cubic initial conditions
    U0 = 1.6
    pmin = -2
    pmax = 2
    pstep = 100

    # par_list, solutions = nat_continuation(cubic_eq, U0, pmin, pmax, pstep, discretisation=lambda x: x)
    # plt.plot(par_list, solutions)
    # plt.show()

    U0 = 1.4, 0, 6.3
    pmin = 2
    pmax = 0
    pstep = 50

    # par_list, solutions = nat_continuation(hopfn, U0, pmin, pmax, pstep, shooting)
    # plt.plot(par_list, solutions)
    # plt.show()

    pmin = 2
    pmax = -1
    pstep = 34

    # par_list, solutions = nat_continuation(hopfm, U0, pmin, pmax, pstep, shooting)
    # plt.plot(par_list, solutions)
    # plt.show()

    # U0 = 1.4, 0, 6.3
    # params = [2, -1]
    # pmin = -1
    # pmax = 2
    # pstep = 0.1
    #
    # par_list, solutions = nat_continuation_h(hopf, U0, params, pmin, pmax, pstep, 0, shooting)

    U0 = 1.6
    pmin = -2
    pmax = 2
    pstep = 100

    val1, val2 = psuedo_continuation(cubic_eq, U0, pmin, pmax, pstep, discretisation=lambda x: x)

    # plt.plot(par_list, solutions)
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
        return 5


    # had to make function calling pde solver with 2 arguments
    def pdef(U0, arg):

        args = [arg, 1, 0.5]
        u_j = solve_pde(mx, mt, 'CN', 'dirichlet', p, q, args)

        return u_j


    def cont_pde(f, U0, args):
        return f(U0, args)


    # param, sols = nat_continuation(pdef, np.zeros(mx+1), 0.5, 2, 11, lambda x: x, 'cont_pde')
    # print(sols[-1], 'solutions')
    # t = np.linspace(0, T, mx + 1)  # mesh points in time
    #
    # j = 0
    #
    # for i in sols:
    #     ka = np.round(param[j], 4)
    #     ka = str(ka)
    #     plt.plot(t, i, label='T = ' + ka)
    #     plt.legend()
    #     j += 1
    # plt.show()
    # solve_pde(mx, mt, 'CN', 'dirichlet', p, q, 2)
