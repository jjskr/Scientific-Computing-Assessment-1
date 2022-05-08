import numpy as np
from solve_odes import f2
from solve_odes import f
from solve_odes import solve_ode
from solve_odes import solve_to
from solve_odes import euler_step
from solve_odes import runge_kutta
from num_shooting import orbit
from num_shooting import shooting
from scipy.optimize import fsolve


def code_testing_solve_ode():

    # testing for incorrect inputs

    suitable_x0 = 0
    suitable_t0 = 0
    suitable_t1 = 10
    suitable_dt = 0.01
    suitable_method = 'runge'

    unsuitable_x0 = 'str'
    long_x0_f = 1, 1
    unsuitable_t0 = 'str'
    unsuitable_t1 = 'str'
    unsuitable_dt = 'str'
    unsuitable_method = 10
    unsuitable_eq = 10

    tests_passed = []
    tests_failed = []

    try:
        solve_ode(unsuitable_x0, suitable_t0, suitable_t1, f, suitable_method, suitable_dt)
        tests_failed = tests_failed + ['unsuitable initial conditions type test']
    except TypeError:
        tests_passed = tests_passed + ['unsuitable initial conditions type test']

    try:
        solve_ode(suitable_x0, unsuitable_t0, suitable_t1, f, suitable_method, suitable_dt)
        tests_failed = tests_failed + ['unsuitable initial time type test']
    except TypeError:
        tests_passed = tests_passed + ['unsuitable initial time type test']

    try:
        solve_ode(suitable_x0, suitable_t0, unsuitable_t1, f, suitable_method, suitable_dt)
        tests_failed = tests_failed + ['unsuitable final time type test']
    except TypeError:
        tests_passed = tests_passed + ['unsuitable final time type test']

    try:
        solve_ode(suitable_x0, suitable_t0, suitable_t1, unsuitable_eq, suitable_method, suitable_dt)
        tests_failed = tests_failed + ['unsuitable function type test']
    except TypeError:
        tests_passed = tests_passed + ['unsuitable function type test']

    try:
        ans = solve_ode(long_x0_f, suitable_t0, suitable_t1, f, suitable_method, suitable_dt)
        try:
            list(ans)
            tests_failed = tests_failed + ['args needed/unsuitable initial conditions length test']
        except TypeError:
            tests_passed = tests_passed + ['args needed/unsuitable initial conditions length test']
    except (IndexError, TypeError):
        tests_passed = tests_passed + ['args needed/unsuitable initial conditions length test']

    try:
        solve_ode(suitable_x0, suitable_t0, suitable_t1, f, unsuitable_method, suitable_dt)
        tests_failed = tests_failed + ["unsuitable method type test"]
    except TypeError:
        tests_passed = tests_passed + ["unsuitable method type test"]

    try:
        solve_ode(suitable_x0, suitable_t0, suitable_t1, f, suitable_method, unsuitable_dt)
        tests_failed = tests_failed + ['unsuitable dt type test']
    except ValueError:
        tests_passed = tests_passed + ['unsuitable dt type test']

    return tests_failed, tests_passed


def code_testing_orbit_function():

    suitable_U0 = 1.5, 1.5, 5
    suitable_ode = hopf
    suitable_pc = pc_stable_0
    suitable_args = [1, -1]

    unsuitable_U0 = 'str'
    unsuitable_ode = 'str'
    unsuitable_pc = 'str'
    unsuitable_args = 0




def test_input_orbit_shooting(ode, U0, pc, *args):
    """
    Function to test my input into my orbit function
    :param ode: ODE(s) to solve for
    :param U0: initial conditions (including time)
    :param args: additional arguments to pass to ODE(s)
    :return: nothing if tests pass, error message if test fails
    """

    try:
        ini_cons = len(U0)
    except TypeError:
        raise TypeError('Initial conditions wrong type')

    if ini_cons < 1:
        raise IndexError('Initial conditions too short')
    else:
        u0, t0 = U0[:-1], U0[-1]

    try:
        is_fun = str(ode)[1]
        if is_fun == 'f':
            pass
        else:
            raise TypeError('Given ode(s) not a function')
    except (IndexError, TypeError):
        raise TypeError('Given ode(s) not a function')

    try:
        is_fun = str(pc)[1]
        if is_fun == 'f':
            pass
        else:
            raise TypeError('Given phase condition not a function')
    except (IndexError, TypeError):
        raise TypeError('Given phase condition not a function')

    try:
        ode(u0, t0, *args)
        print('ode solves for initial conditions')
    except IndexError:
        print('args not suitable')
        return
    except TypeError:
        print('extra initial condition/args needed')
        return
    except ValueError:
        print('more initial conditions needed')

    try:
        is_fun = str(shooting)[1]
        if is_fun == 'f':
            pass
    except (IndexError, TypeError):
        print('shooting method not a function')
        return

    if args:
        args = (pc, *args)
        sol = fsolve(shooting(ode), U0, args)
    else:
        sol = fsolve(shooting(ode), U0, pc)
    print(sol, 'sol')


def test_input_solve_ode(x0, t0, t1, eqs, method, deltat_max, *args):
    """
    Function used to test inputs to solve odes
    :param x0: initial conditions
    :param t0: initial time
    :param t1: final time
    :param eqs: ODE(s) to solve
    :param method: step method to use
    :param deltat_max: max step size
    :param args: additional arguments needed by ODE(s)
    :return: nothing if tests pass, message if a test fails
    """

    if args:
        args = args[0]
        print(args)
        try:
            eqs(x0, t0, args)
            print('args suitable')

        except (TypeError, IndexError):
            print('args not suitable')
            return
    else:
        try:
            ret = eqs(x0, t0)
            if isinstance(ret, int) or isinstance(ret, list):
                print('equation and inputs suitable')
            else:
                err = 5
                return err
        except (TypeError, IndexError):
            print('args needed')
            return


if __name__ == '__main__':

    def hopf(U0, t, args):
        """
        A function that returns solutions to the hopf equations at (U0, t)
        :param U0: values for x and y
        :param t: time
        :param args: list of beta and sigma constants
        :return: solutions to hopf equations
        """
        # print(U0, t)
        beta = args[0]
        sigma = args[1]
        u1, u2 = U0
        du1dt = beta * u1 - u2 + sigma * u1 * (u1 ** 2 + u2 ** 2)
        du2dt = u1 + beta * u2 + sigma * u2 * (u1 ** 2 + u2 ** 2)

        return [du1dt, du2dt]

    def pc_stable_0(U0, T, ode, args):
        return ode(U0, 0, args)[0]


    tests_failed, tests_passed = code_testing_solve_ode()

    print(tests_failed)
    print(tests_passed)
    # method = 'runge'
    eqs = hopf
    ode = hopf
    # args = [1, -1]
    # x0 = 1.5, 1.5
    # t0 = 5
    # deltat_max = 0.01
    # test_input_solve_ode(x0, t0, t1, eqs, method, deltat_maxx, args)
    #
    #
    U0 = 1.5, 1.5, 5
    # U0 = 1.5, 1.5, -700
    args = [1, -1]
    # args = [1]
    # test_input_orbit_shooting(hopf, U0, pc_stable_0, args)
    # orb = orbit(hopf, U0, pc_stable_0, args)
    # solve_ode(unsuitable_x0, suitable_t0, suitable_t1, f, suitable_method, suitable_dt)
