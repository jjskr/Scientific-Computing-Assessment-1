from scipy.optimize import fsolve
from num_shooting import shooting
import matplotlib.pyplot as plt
# from num_shooting import hopf
import numpy as np
from pde_solving import solve_pde
from math import pi
import warnings

warnings.filterwarnings('ignore')


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

    # rounding can be more lenient with pde calculations
    if solver == 'pde':
        rou = 10
        solver = cont_pde_solver
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

    def p_upd(pred):
        """
        Function to update parameter in pseudo arclength equation
        :param pred: next prediction
        :return: par updated as prediction
        """

        par = pred

        return par

    params = np.linspace(par_min, par_max, par_split)

    diff = params[1] - params[0]
    par0 = params[0]
    par1 = par0 + diff

    param_l = [par0, par1]

    if discretisation == shooting:
            par0 = [pc_stable_0, par0]
            par1 = [pc_stable_0, par1]

    val0 = fsolve(discretisation(ode), U0, args=par0)
    val1 = fsolve(discretisation(ode), val0, args=par1)

    sol_lis = [val0[0], val1[0]]

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

        sol = solver(lambda cur_s: np.append(discretisation(ode)(cur_s[:-1], p_upd(cur_s[-1])), np.dot(cur_s[:-1] - pred_x, delta_x) + np.dot(cur_s[-1] - pred_p, delta_p)), pred_ar)

        sol_add = sol[:-1][0]
        par_add = sol[-1]
        sol_lis = sol_lis + [sol_add]
        param_l = param_l + [par_add]

        i += 1

    return sol_lis, param_l


def psuedo_continuation_h(ode, U0, par_min, par_max, par_split, discretisation, solver=fsolve):
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

    def p_upd(pred):
        """
        Function to update parameter in pseudo arclength equation
        :param pred: next prediction
        :return: par updated as prediction
        """

        par = pred

        return par

    params = np.linspace(par_min, par_max, par_split)

    diff = params[1] - params[0]
    par0 = params[0]
    par1 = par0 + diff

    param_l = [par0, par1]

    print(param_l)

    if discretisation == shooting:
            par0 = [pc_stable_0, par0]
            par1 = [pc_stable_0, par1]

    val0 = fsolve(discretisation(ode), U0, args=par0)
    val1 = fsolve(discretisation(ode), val0, args=par1)

    print(val0, val1)

    sol_lis = [val0[0], val1[0]]

    print(sol_lis)

    i = 0

    par_add = par_max

    print(par_max, par_min)

    while par_add < par_min:

        # generate sec
        delta_x = sol_lis[i+1] - sol_lis[i]
        delta_p = param_l[i+1] - param_l[i]

        pred_x = sol_lis[i+1] + delta_x
        pred_p = param_l[i+1] + delta_p

        pred_ar = [pred_x, pred_p]

        pred_ar = np.array(pred_ar)

        if discretisation == shooting:
            sol = solver(lambda cur_s: np.append(discretisation(ode)(cur_s[:-1], [pc_stable_0, p_upd(cur_s[-1])]),
                                                 np.dot(cur_s[:-1] - pred_x, delta_x) + np.dot(cur_s[-1] - pred_p,
                                                                                               delta_p)), pred_ar)

        else:
            sol = solver(lambda cur_s: np.append(discretisation(ode)(cur_s[:-1], p_upd(cur_s[-1])),
                                                    np.dot(cur_s[:-1] - pred_x, delta_x) + np.dot(cur_s[-1] - pred_p,
                                                                                                  delta_p)), pred_ar)
        #sol = solver(lambda cur_s: np.append(discretisation(ode)(cur_s[:-1], p_upd(cur_s[-1])),
                                             # np.dot(cur_s[:-1] - pred_x, delta_x) + np.dot(cur_s[-1] - pred_p,
                                             #                                               delta_p)), pred_ar)
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
    :param discretisation: method of discretisation - 'shooting' or lambda x: x
    :param solver: solver defaults as fsolve, cont_pde for pdes
    :return: solutions for parameters in parameter list
    """

    if isinstance(U0, tuple) or isinstance(U0, int):
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

    try:
        is_fun = str(ode)[1]
        if is_fun == 'f':
            pass
        else:
            raise TypeError('Given ode not a function')
    except IndexError:
        raise TypeError('Given ode not a function')

    if pc:
        try:
            is_fun = str(pc)[1]
            if is_fun == 'f':
                pass
            else:
                raise TypeError('Given phase condition not a function')
        except IndexError:
            raise TypeError('Given phase condition not a function')

    if method == 'pseudo':
        sols, pars = psuedo_continuation(ode, U0, par_min, par_max, par_split, discretisation, solver)
    elif method == 'natural':
        sols, pars = nat_continuation(ode, U0, par_min, par_max, par_split, pc, discretisation, solver)
    else:
        raise ValueError('method not suitable')

    return sols, pars


if __name__ == '__main__':

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
    #
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

    def ode_equivalent(U0, arg):
        """
        Function to return the function for solver to find root of
        :param U0: initial conditions (unused)
        :param arg: time value (can be changed but have to change index of 'arg' in args below)
                    index: 0 = kappa, 1 = length, 2 = time
        :return: state of PDE at chosen time
        """
        # can vary changing parameter by change position of arg in list below
        # very tidious would change this with more time
        args = [1, 1, arg]
        u_j = solve_pde(mx, mt, 'CN', 'dirichlet', p, q, False, args)

        return u_j


    def cont_pde_solver(f, U0, args):
        """
        New solver defined as fsolve is not suitable, we need to pass on to solve_pde instead, function will
        pass U0 and args into pdef, updating args and calling solve_pde
        :param f:
        :param U0:
        :param args:
        :return:
        """
        return f(U0, args)

    # define homogeneous boundary conditions

    def p(t):
        """
        LHS boundary condition
        :param t: time value
        :return: boundary condition at t
        """
        return 0


    def q(t):
        """
        RHS boundary condition
        :param t: time value
        :return: boundary condition at t
        """
        return 0

    mx = 30  # number of gridpoints in space
    mt = 1000  # number of gridpoints in time

    U0 = tuple(np.zeros(31))
    param, sols = continuation('natural', ode_equivalent, U0, 0, 0.5, 5, None, lambda x: x, 'pde')
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

    # solve_pde(mx, mt, 'CN', 'dirichlet', p, q, 2)
