import numpy as np
from solve_odes import f2
from solve_odes import f
from solve_odes import solve_ode
from solve_odes import solve_to
from solve_odes import euler_step
from solve_odes import runge_kutta
from num_shooting import hopf


def test_input_solve_ode(x0, t0, t1, eqs, method, deltat_max, *args):

    if isinstance(x0, tuple) or isinstance(x0, int):
        print('starting co-ordinate(s) suitable type')
    else:
        print('starting co-ordinate(s) wrong type')

    if isinstance(t0, int):
        print('initial time suitable type')
    else:
        print('initial time wrong type')

    if isinstance(t1, int):
        print('final time suitable type')
    else:
        print('final time wrong type')

    try:
        is_fun = str(eqs)[1]
        if is_fun == 'f':
            print('system of odes suitable')
    except IndexError:
        print('system of odes not suitable')

    try:
        eqs(x0, t0, *args)
        print('initial conditions suitable')
    except TypeError:
        print('initial conditions not suitable')

    if method == 'runge' or method == 'euler':
        print('method suitable')
    else:
        print('method unsuitable')

    try:
        float(deltat_max)
        print('delta t suitable')
    except ValueError:
        print('delta t not suitable')

    if args:
        args = args[0]
        try:
            eqs(x0, t0, args)
            print('args suitable')

        except (TypeError, IndexError):
            print('args not suitable')
    else:
        try:
            eqs(x0, t0)
            print('equation and inputs suitable')
        except (TypeError, IndexError):
            print('args needed')


t0 = 0
t1 = 10
x0 = 1, 0
deltat_maxx = 0.1
method = 'runge'
eqs = f2
test_input_solve_ode(x0, t0, t1, eqs, method, deltat_maxx)


method = 'runge'
eqs = hopf
args = [1, -1]
x0 = 1.5, 1.5
t0 = 5
deltat_max = 0.01
test_input_solve_ode(x0, t0, t1, eqs, method, deltat_maxx, args)
