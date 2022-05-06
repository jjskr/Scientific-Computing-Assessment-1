import numpy as np
from solve_odes import f2
from solve_odes import f
from solve_odes import solve_ode
from solve_odes import solve_to
from solve_odes import euler_step
from solve_odes import runge_kutta
from num_shooting import hopf
from num_shooting import orbit
from num_shooting import shooting
from scipy.optimize import fsolve


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
    :return: nothing if tests pass, message if not
    """
    if isinstance(x0, tuple) or isinstance(x0, int):
        print('starting co-ordinate(s) suitable type')
    else:
        print('starting co-ordinate(s) wrong type')
        return

    if isinstance(t0, int) or isinstance(t0, float):
        print('initial time suitable type')
    else:
        print('initial time wrong type')
        return

    if isinstance(t1, int) or isinstance(t1, float):
        print('final time suitable type')
    else:
        print('final time wrong type')
        return
    try:
        is_fun = str(eqs)[1]
        if is_fun == 'f':
            print('system of odes suitable')
    except IndexError:
        print('system of odes not suitable')
        return

    if args:
        args = args[0]
        try:
            eqs(x0, t0, args)
            print('args suitable')

        except (TypeError, IndexError):
            print('args not suitable')
            return
    else:
        try:
            eqs(x0, t0)
            print('equation and inputs suitable')
        except (TypeError, IndexError):
            print('args needed')
            return

    try:
        eqs(x0, t0, *args)
        print('initial conditions suitable')
    except TypeError:
        print('initial conditions/args not suitable')
        return

    if method == 'runge' or method == 'euler':
        print('method suitable')
    else:
        print('method unsuitable')
        return
    try:
        float(deltat_max)
        print('delta t suitable')
    except ValueError:
        print('delta t not suitable')
        return


t0 = 0
t1 = 10
x0 = 1, 0
deltat_maxx = 0.1
method = 'runge'
eqs = f2
# test_input_solve_ode(x0, t0, t1, eqs, method, deltat_maxx)


method = 'runge'
eqs = hopf
# args = [1, -1]
x0 = 1.5, 1.5
t0 = 5
deltat_max = 0.01
# test_input_solve_ode(x0, t0, t1, eqs, method, deltat_maxx, args)


def test_input_orbit_shooting(ode, U0, *args):
    """
    Function to test my input into my orbit function
    :param ode: ODE(s) to solve for
    :param U0: initial conditions (including time)
    :param args: additional arguments to pass to ODE(s)
    :return: nothing if tests pass, error message if tests fail
    """

    try:
        ini_cons = len(U0)
    except TypeError:
        print('initial conditions wrong type')
        return

    if ini_cons > 1:
        u0, t0 = U0[:-1], U0[-1]
    else:
        print('initial conditions too short')
        return

    try:
        is_fun = str(ode)[1]
        if is_fun == 'f':
            pass
    except (IndexError, TypeError):
        print('system of odes not suitable')
        return
    # print(*args)
    # print(ode(u0, t0, *args))

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

    sol = fsolve(shooting(ode), U0, *args)
    print(sol)


U0 = 1.5, 1.5, 5
# U0 = 1.5, 1.5
args = [1, -1]
# args = [1]
test_input_orbit_shooting(hopf, U0, args)