from solve_odes import solve_ode
from num_shooting import orbit


def code_testing_solve_ode():

    # testing for incorrect inputs

    suitable_x0 = 0
    suitable_t0 = 0
    suitable_t1 = 10
    suitable_dt = 0.01
    suitable_method = 'runge'

    unsuitable_x0 = 'str'
    long_x0_f = (1, 1)
    unsuitable_t0 = 'str'
    unsuitable_t1 = 'str'
    unsuitable_dt = 'str'
    unsuitable_method = 10
    unsuitable_eq = 10

    tests_passed = []
    tests_failed = []

    # test initial conditions dimensions

    try:
        solve_ode(long_x0_f, suitable_t0, suitable_t1, f, suitable_method, suitable_dt)
        tests_failed = tests_failed + ['unsuitable initial condition dimensions']
    except TypeError:
        tests_passed = tests_passed + ['unsuitable initial condition dimensions']

    # test initial conditions type
    try:
        solve_ode(unsuitable_x0, suitable_t0, suitable_t1, f, suitable_method, suitable_dt)
        tests_failed = tests_failed + ['unsuitable initial conditions type test']
    except TypeError:
        tests_passed = tests_passed + ['unsuitable initial conditions type test']

    # test initial time type
    try:
        solve_ode(suitable_x0, unsuitable_t0, suitable_t1, f, suitable_method, suitable_dt)
        tests_failed = tests_failed + ['unsuitable initial time type test']
    except TypeError:
        tests_passed = tests_passed + ['unsuitable initial time type test']

    # test final time type
    try:
        solve_ode(suitable_x0, suitable_t0, unsuitable_t1, f, suitable_method, suitable_dt)
        tests_failed = tests_failed + ['unsuitable final time type test']
    except TypeError:
        tests_passed = tests_passed + ['unsuitable final time type test']

    # test ode type
    try:
        solve_ode(suitable_x0, suitable_t0, suitable_t1, unsuitable_eq, suitable_method, suitable_dt)
        tests_failed = tests_failed + ['unsuitable function type test']
    except TypeError:
        tests_passed = tests_passed + ['unsuitable function type test']

    # test method type
    try:
        solve_ode(suitable_x0, suitable_t0, suitable_t1, f, unsuitable_method, suitable_dt)
        tests_failed = tests_failed + ["unsuitable method type test"]
    except TypeError:
        tests_passed = tests_passed + ["unsuitable method type test"]

    # test dt type
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
    unsuitable_U0_L = 1, 1
    unsuitable_ode = 'str'
    unsuitable_pc = 'str'
    unsuitable_args = 0

    tests_failed = []
    tests_passed = []

    # test initial condition dimensions
    try:
        orbit(suitable_ode, unsuitable_U0_L, suitable_pc, suitable_args)
        tests_failed = tests_failed + ['unsuitable initial condition dimensions']
    except TypeError:
        tests_passed = tests_passed + ['unsuitable initial condition dimensions']

    # test ode type
    try:
        orbit(unsuitable_ode, suitable_U0, suitable_pc, suitable_args)
        tests_failed = tests_failed + ['unsuitable ode test']
    except TypeError:
        tests_passed = tests_passed + ['unsuitable ode test']

    # testing initial conditions type
    try:
        orbit(suitable_ode, unsuitable_U0, suitable_pc, suitable_args)
        tests_failed = tests_failed + ['unsuitable initial conditions test']
    except TypeError:
        tests_passed = tests_passed + ['unsuitable initial conditions test']

    try:
        orbit(suitable_ode, suitable_U0, unsuitable_pc, suitable_args)
        tests_failed = tests_failed + ['unsuitable phase condition test']
    except TypeError:
        tests_passed = tests_passed + ['unsuitable phase condition test']

    return tests_failed, tests_passed


if __name__ == '__main__':

    def f(x, t):
        """
        A function that returns the value of dxdt = x at (x, t)
        :param x: x value
        :param t: t value
        :return: value of ODE at (x, t)
        """
        return x

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

    def pc_stable_0(U0, T, ode, args):
        return ode(U0, 0, args)[0]


    tests_failed_s, tests_passed_s = code_testing_solve_ode()
    tests_failed_o, tests_passed_o = code_testing_orbit_function()

    print('--------------------------')
    print('SOLVE ODE ')
    print('TESTS FAILED: ', tests_failed_s)
    print('TESTS PASSED: ', tests_passed_s)
    print('--------------------------')
    print('ORBIT SHOOTING')
    print('TESTS FAILED: ', tests_failed_o)
    print('TESTS PASSED: ', tests_passed_o)
    print('--------------------------')

    # dimension test fails for solve ode :(
