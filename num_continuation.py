from scipy.optimize import fsolve
from num_shooting import shooting
import matplotlib.pyplot as plt
# from num_shooting import hopf
import numpy as np
from pde_solving import solve_pde
from math import pi
import warnings

warnings.filterwarnings('ignore')


def nat_continuation(ode, U0, params_list, pc, discretisation, solver=fsolve):
    """
    Performs natural parameter continuation on chosen set of ODEs returning parameter list and corresponding solution
    for each parameter value

    :param ode: System of ODEs to solve
    :param U0: Initial conditions
    :param params_list: List of values to use for varying parameter
    :param discretisation: Discretisation to use to solve
    :param solver: Type of solver to use. (only implemented fsolve and cont_pde for pdes)
    :return: List of parameters and solutions at each parameter value
    """

    solut_list = []

    for par in params_list:
        if pc:
            args = (pc, par)
        else:
            args = par
        if solver == 'pde':
            # limited to dirichlet homogeneous boundaries, potential improvement
            U0 = solve_pde(30, 1000, 'CN', 'dirichlet', lambda x: 0, lambda x: 0, False, [1, 1, par])
        else:
            U0 = solver(discretisation(ode), U0, args)
        solut_list = solut_list + [U0]
        U0 = np.round(U0, 5)

    return params_list, solut_list


def psuedo_continuation(ode, U0, params_list, discretisation):
    """
    Performs pseudo-arclength continuation on chosen set of ODEs returning parameter list and
    corresponding solution for each parameter value
    :param ode: System of ODEs to solve
    :param U0: Initial conditions
    :param params_list: List of parameters to use
    :param discretisation: Discretisation to use to solve
    :return: List of parameters and solutions at each parameter value
    """

    def p_upd(pred):
        """
        Function to update parameter in pseudo arclength equation
        :param pred: next prediction
        :return: par updated as prediction
        """
        par = pred
        return par

    # first two parameter values calculated
    diff = params_list[1] - params_list[0]
    par0 = params_list[0]
    par1 = par0 + diff
    param_l = [par0, par1]

    if discretisation == shooting:
            par0 = [pc_stable_0, par0]
            par1 = [pc_stable_0, par1]

    # initial two values found to start pseudo-arclength continuation
    val0 = fsolve(discretisation(ode), U0, args=par0)
    val1 = fsolve(discretisation(ode), val0, args=par1)
    sol_lis = [val0[0], val1[0]]
    i = 0

    # while loop iterates through until final parameter is reached
    par_add = params_list[0]
    while par_add < params_list[-1]:

        # generate sec
        delta_x = sol_lis[i+1] - sol_lis[i]
        delta_p = param_l[i+1] - param_l[i]
        # prediction for next value updates
        pred_x = sol_lis[i+1] + delta_x
        pred_p = param_l[i+1] + delta_p
        pred_ar = [pred_x, pred_p]
        pred_ar = np.array(pred_ar)
        # solution found by appending
        sol = fsolve(lambda cur_s: np.append(discretisation(ode)(cur_s[:-1], p_upd(cur_s[-1])), np.dot(cur_s[:-1] - pred_x, delta_x) + np.dot(cur_s[-1] - pred_p, delta_p)), pred_ar)
        # solution array updated
        sol_add = sol[:-1][0]
        par_add = sol[-1]
        sol_lis = sol_lis + [sol_add]
        param_l = param_l + [par_add]

        i += 1

    return sol_lis, param_l


def continuation(method, ode, U0, par_min, par_max, par_split, pc, discretisation, solver=fsolve):
    """
    Continuation function passing parameters onto desired method of continuation
    :param method: choice between 'pseudo' and 'normal'
    :param ode: system of ODEs to perform chosen continuation on
    :param U0: initial conditions
    :param par_min: parameter to begin at
    :param par_max: parameter to finish at
    :param par_split: number of splits in parameter list
    :param pc: phase condition - None if not required
    :param discretisation: method of discretisation to use to solve - 'shooting' or lambda x: x
    :param solver: solver defaults as fsolve, cont_pde for pdes
    :return: solutions for parameters in parameter list
    """

    if isinstance(U0, tuple) or isinstance(U0, int) or not U0:
        pass
    else:
        raise TypeError('Initial conditions wrong type')

    if isinstance(par_min, int) or isinstance(par_min, float):
        pass
    else:
        raise TypeError('Initial parameter value wrong type')

    if isinstance(par_max, int) or isinstance(par_max, float):
        pass
    else:
        raise TypeError('Final parameter value wrong type')

    if isinstance(par_split, int):
        pass
    else:
        raise TypeError('Number of splits wrong type')

    params_list = np.linspace(par_min, par_max, par_split)

    if method == 'pseudo':
        sols, pars = psuedo_continuation(ode, U0, params_list, discretisation)
    elif method == 'natural':
        sols, pars = nat_continuation(ode, U0, params_list, pc, discretisation, solver)
    else:
        raise ValueError('method not suitable')

    return sols, pars


if __name__ == '__main__':

    def pc_stable_0(U0, T, ode, args):
        """
        Phase condition for hopf
        :param U0: current conditions
        :param T: current time
        :param ode: system of equations to solve
        :param args: additional arguments to pass to function
        :return: result of first equation in system of odes for U0
        """
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
        cubic_sol = x ** 3 - x + c

        return cubic_sol


    def hopfn(U0, t, args):

        beta = args

        u1, u2 = U0

        du1dt = beta * u1 - u2 - u1 * (u1 ** 2 + u2 ** 2)
        du2dt = u1 + beta * u2 - u2 * (u1 ** 2 + u2 ** 2)

        return [du1dt, du2dt]


    def hopfm(U0, t, args):

        beta = args

        u1, u2 = U0

        du1dt = beta * u1 - u2 + u1 * (u1 ** 2 + u2 ** 2) - u1 * (u1 ** 2 + u2 ** 2) ** 2
        du2dt = u1 + beta * u2 + u2 * (u1 ** 2 + u2 ** 2) - u2 * (u1 ** 2 + u2 ** 2) ** 2

        return [du1dt, du2dt]

    # experimenting

    U0 = 1.4, 0, 6.3
    pmin = 2
    pmax = -1
    pstep = 100

    # sol_l, p_l = psuedo_continuation_h(hopfn, U0, pmin, pmax, pstep, shooting)
    # plt.plot(p_l, sol_l)
    # plt.show()

    # plotting natural continuation and pseudo arc length results for cubic
    # as c varies between -2 and 2
    # cubic initial conditions
    U0 = 1
    pmin = -2
    pmax = 2
    pstep = 100
    #
    par_list, solutions = continuation('natural', cubic_eq, U0, pmin, pmax, pstep, None, lambda x: x)
    plt.plot(par_list, solutions, label='natural parameter')
    # #
    U0 = 1
    pmin = -2
    pmax = 2
    pstep = 100

    sol_l, p_l = continuation('pseudo', cubic_eq, U0, pmin, pmax, pstep, None, lambda x: x)
    plt.plot(p_l, sol_l, label='pseudo-arclength')
    plt.title('Plot showing performance of continuation methods on cubic equation')
    plt.legend()
    plt.show()

    # # hopf continuation - natural works but could not get pseudo arclength to work

    U0 = 1.4, 0, 6.3
    pmin = 2
    pmax = 0
    pstep = 50

    par_list, solutions = continuation('natural', hopfn, U0, pmin, pmax, pstep, pc_stable_0, shooting)
    U0_sols = []
    for i in solutions:
        U0_sols = U0_sols + [i[0]]
    plt.plot(par_list, U0_sols)
    plt.title('Plot showing performance of natural continuation on Hopfield equations')
    plt.show()
    #
    # # modified hopf plots
    #
    pmin = 2
    pmax = -1
    pstep = 34

    par_list_m, solutions_m = continuation('natural', hopfm, U0, pmin, pmax, pstep, pc_stable_0, shooting)
    U0_sols_m = []
    for i in solutions_m:
        U0_sols_m = U0_sols_m + [i[0]]
    plt.plot(par_list_m, U0_sols_m)
    plt.title('Plot showing performance of natural continuation on modified Hopfield equations')
    plt.show()

    # pde continuation, varying t between 0 and 0.5 with homogeneous boundaries
    # shows the evolution of the gridspace as t varies

    mx = 30  # number of gridpoints in space
    mt = 1000  # number of gridpoints in time

    param, sols = continuation('natural', None, None, 0, 0.5, 10, None, lambda x: x, 'pde')

    t = np.linspace(0, 0.5, mx + 1)  # mesh points in time

    j = 0

    for i in sols:
        ka = np.round(param[j], 4)
        ka = str(ka)
        plt.plot(t, i, label='T = ' + ka)
        plt.legend()
        j += 1

    plt.title('Plot showing the state of the grid as time varies')
    plt.show()
