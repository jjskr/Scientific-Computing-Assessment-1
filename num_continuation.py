from scipy.optimize import fsolve
from num_shooting import shooting
import matplotlib.pyplot as plt
import numpy as np
from pde_solving import solve_pde
from num_shooting import orbit
import warnings
import time
# ignore runtime warnings caused by natural parameter continuation
warnings.filterwarnings('ignore')


def nat_continuation(ode, U0, params_list, pc, discretisation, solver):
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
        if discretisation == shooting:
            args = (pc, par)
        else:
            args = par
        if solver == solve_pde:
            # limited to dirichlet homogeneous boundaries, potential improvement
            # also only allows time to be varied
            U0 = solver(30, 1000, 'CN', 'dirichlet', lambda x: 0, lambda x: 0, False, [1, 1, par])
        else:
            U0 = solver(discretisation(ode), U0, args)
        solut_list = solut_list + [U0]
        # rounding needed for Hopfield equations or solve_ode breaks
        if discretisation == shooting:
            U0 = np.round(U0, 6)

    return solut_list, params_list


def psuedo_continuation(ode, U0, params_list, discretisation, pc):
    """
    Performs pseudo-arclength continuation on chosen set of ODEs returning parameter list and
    corresponding solution for each parameter value

    :param ode: System of ODEs to solve
    :param U0: Initial conditions
    :param params_list: List of parameters
    :param discretisation: Discretisation to use
    :param pc: Phase condition
    :return: List of parameters and solutions at each parameter value
    """
    def new_predict_calc(solution_list, parameter_list, index):
        """
        Function returns updated predictions for pseudo arclength
        :param solution_list: list of solutions for given iteration
        :param parameter_list: list of parameters for given iteration
        :param index: current iteration
        :return: delta x, delta p and new prediction array
        """
        x_change = solution_list[index + 1] - solution_list[index]
        p_change = parameter_list[index + 1] - parameter_list[index]
        pred_arr = np.r_[solution_list[index + 1] + x_change, parameter_list[index + 1] + p_change]

        return x_change, p_change, pred_arr

    def pseudo_eq(x, prediction_array):
        """
        Function returning pseudo-arclength equation at each iteration
        :param x: state vector to solve for
        :param prediction_array: array containing prediction information
        :return: pseudo-arclength equation
        """
        return np.dot(delta_x, x[:-1] - prediction_array[:-1]) + np.dot(delta_p, x[-1] - prediction_array[-1])
    # adds phase condition to args if shooting
    if discretisation == shooting:
        args = (pc, params_list[0])
        args1 = (pc, params_list[1])
        u0 = fsolve(discretisation(ode), U0, args)
        # rounding needed or error in solve ode, potential improvement
        val_round = np.round(u0, 5)
        u1 = fsolve(discretisation(ode), val_round, args1)
    else:
        u0 = fsolve(discretisation(ode), U0, params_list[0])
        u1 = fsolve(discretisation(ode), u0, params_list[1])

    sol_lis, param_l = [u0, u1], [params_list[0], params_list[1]]

    # limited iterations due to length of time taken to run for hopf equations
    # chose high enough i value to show pseudo-arclength method making it around the curve
    for i in range(80):
        # updates parameters to pass into equation
        delta_x, delta_p, prediction_array = new_predict_calc(sol_lis, param_l, i)
        # equation updated until root is found,
        if discretisation == shooting:
            # root found for solving discretisation problem with updated pseudo arclength equation
            # allows parameter value to be solved for to trace around curves after equilibrium
            sol = fsolve(lambda x: np.append(discretisation(ode)(x[:-1], pc, x[-1]), pseudo_eq(x, prediction_array)),
                         prediction_array)
        # pc not needed for cubic equation
        else:
            sol = fsolve(lambda x: np.append(discretisation(ode)(x[:-1], x[-1]), pseudo_eq(x, prediction_array)),
                         prediction_array)
        # giving updates in terminal if shooting is used
        if discretisation == shooting:
            if i == 40:
                print('Approximately halfway')
        # solution and parameter values added to respective lists
        sol_lis, param_l = sol_lis + [sol[:-1]], param_l + [sol[-1]]

    return sol_lis, param_l


def continuation(method, ode, U0, par_min, par_max, par_split, pc, discretisation, solver):
    """
    Continuation function passing parameters onto desired method of continuation
    :param method: choice between 'pseudo' and 'normal'
    :param ode: system of ODEs to perform chosen continuation on
    :param U0: initial conditions
    :param par_min: parameter to begin at, if shooting then initial time
    :param par_max: parameter to finish at
    :param par_split: number of splits in parameter list
    :param pc: phase condition - None if not required
    :param discretisation: method of discretisation to use to solve - 'shooting' or lambda x: x
    :param solver: solver defaults as fsolve, cont_pde for pdes
    :return: solutions for parameters in parameter list
    """
    # input error traps
    if isinstance(U0, tuple) or isinstance(U0, int) or not U0:
        pass
    else:
        raise TypeError('Initial conditions wrong type')

    if isinstance(par_min, int) or isinstance(par_min, float):
        pass
    else:
        raise TypeError('Initial parameter value wrong type')

    if isinstance(par_max, int) or isinstance(par_max, float) or not par_max:
        pass
    else:
        raise TypeError('Final parameter value wrong type')

    if isinstance(par_split, int) or not par_split:
        pass
    else:
        raise TypeError('Number of splits wrong type')

    if ode:
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

    if solver == solve_pde or solver == fsolve:
        pass
    else:
        raise ValueError('Given solver not supported')
    # initialise parameter list to pass onto continuation functions
    params_list = np.linspace(par_min, par_max, par_split)

    if method == 'pseudo':
        sols, pars = psuedo_continuation(ode, U0, params_list, discretisation, pc)
        return sols, pars
    elif method == 'natural':
        sols, pars = nat_continuation(ode, U0, params_list, pc, discretisation, solver)
        return sols, pars
    elif method == 'shooting':
        sol = orbit(ode, U0, par_min)
        return sol
    else:
        raise ValueError('method not suitable')


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


    # plotting natural continuation and pseudo arc length results for cubic
    # as c varies between -2 and 2
    # cubic initial conditions

    U0 = 1
    pmin = -2
    pmax = 2
    pstep = 100

    sol_l, p_l = continuation('natural', cubic_eq, U0, pmin, pmax, pstep, None, lambda x: x, fsolve)
    plt.plot(p_l, sol_l, label='natural')
    sol_l, p_l = continuation('pseudo', cubic_eq, U0, pmin, pmax, pstep, None, lambda x: x, fsolve)
    plt.plot(p_l, sol_l, label='pseudo-arclength')
    plt.title('Plot showing performance of continuation methods on cubic equation')
    plt.legend()
    plt.show()

    # hopf continuation - beta varies

    U0 = 1, 1, 6
    pmin = 2
    pmax = -1
    pstep = 50
    # added time simulation for hopf continuation due to long runtimes
    nat_start = time.process_time()
    print('Natural parameter continuation begins')
    solutions, par_list = continuation('natural', hopfn, U0, pmin, pmax, pstep, pc_stable_0, shooting, fsolve)
    nat_end = time.process_time() - nat_start
    print('Done! Time taken: ', nat_end)
    U0_sols = []
    for i in solutions:
        U0_sols = U0_sols + [i[0]]
    plt.plot(par_list, U0_sols)

    pmin = 2
    pmax = -1
    pstep = 70
    param_list = np.linspace(pmin, pmax, pstep)
    pseudo_start = time.process_time()
    print('Pseudo-arclength continuation begins')
    sol_l, p_l = continuation('pseudo', hopfn, U0, pmin, pmax, pstep, pc_stable_0, shooting, fsolve)
    pseudo_taken = time.process_time() - pseudo_start
    print('Done! Time taken: ', pseudo_taken)
    psu_hopf = []
    for i in sol_l:
        psu_hopf = psu_hopf + [i[0]]
    plt.plot(p_l, psu_hopf)
    plt.title('Plot showing continuation results for normal Hopfield equations')
    plt.show()

    # modified hopf plots
    # does not work in pseudo-arclength

    pmin = 2
    pmax = -1
    pstep = 34
    solutions_m, pars_m = continuation('natural', hopfm, U0, pmin, pmax, pstep, pc_stable_0, shooting, fsolve)
    U0_sols_m = []
    for i in solutions_m:
        U0_sols_m = U0_sols_m + [i[0]]
    plt.plot(pars_m, U0_sols_m)
    plt.title('Plot showing natural parameter continuation results for modified Hopfield equations')
    plt.show()

    # pde continuation, varying t between 0 and 0.5 with homogeneous boundaries
    # shows the evolution of the gridspace as t varies

    sols, param = continuation('natural', None, None, 0, 0.5, 10, None, lambda x: x, solve_pde)
    t = np.linspace(0, 0.5, 31)  # mesh points in time
    j = 0
    for i in sols:
        T = round(param[j], 4)
        T = str(T)
        plt.plot(t, i, label='T = ' + T)
        plt.legend()
        j += 1
    plt.title('Plot showing the state of the grid as time varies')
    plt.show()
