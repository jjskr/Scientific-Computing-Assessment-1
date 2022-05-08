import numpy as np
import matplotlib.pyplot as plt
import math
import time


def f(x, t):
    """
    A function that returns the value of dxdt = x at (x, t)
    :param x: x value
    :param t: t value
    :return: value of ODE at (x, t)
    """
    return x


def f2(U0, t):
    """
    A function that returns the value of dxdt = y and dydt = -x at (U0, t)
    :param U0: tuple containing x, y values
    :param t: t value
    :return: values of ODEs at (U0, t)
    """
    x, y = U0
    dxdt = y
    dydt = -x
    return [dxdt, dydt]


def f_true(t):
    """
    A function that returns the true value of dxdt = x at (x, t)
    :param x: x value
    :param t: t value
    :return: true value of ODE at (x, t)
    """
    x = np.exp(t)
    return x


def euler_step(f, x, t, delta_t, *args):
    """
    A function completing 1 euler step, updating x and t
    :param f: either one or a set of ODEs to estimate
    :param x: initial x value
    :param t: initial t value
    :param delta_t: step size
    :param args: additional features/constants to pass into function
    :return: value of x and t after 1 euler step
    """

    x = x + delta_t * np.array(f(x, t, *args))
    t = t + delta_t

    return x, t


def runge_kutta(f, x, t, delta_t, *args):
    """
    A function completing 1 4th order runge-kutta (RK4) step, updating x and t
    :param f: either one or a set of ODEs to estimate
    :param x: initial x value
    :param t: initial t value
    :param delta_t: step size
    :param args: additional features/constants to pass into function
    :return: value of x and t after 1 RK4 step
    """
    k1 = np.array(f(x, t, *args))
    k2 = np.array(f((x + delta_t * k1/2), (t + delta_t/2), *args))
    k3 = np.array(f((x + delta_t * k2/2), (t + delta_t/2), *args))
    k4 = np.array(f((x + delta_t * k3), (t + delta_t), *args))
    k = (k1+2*k2+2*k3+k4)/6
    t = t + delta_t
    x = x + delta_t * k

    return x, t


def solve_to(fun, x0, t0, t1, h, method, *args):
    """
    A function which solves given ODE(s) from initial value (x0, t0) to a given t value using given method and maximum
    step size
    :param fun: ODE(s) to solve
    :param x0: initial x value
    :param t0: initial t value
    :param t1: final t value
    :param h: maximum step size
    :param method: step type - euler or 4th order runge-kutta
    :param args: additional features/constants to pass into function
    :return: x value at t1
    """

    t_diff = t1 - t0

    try:
        intervals = math.floor(t_diff/h)
    except ValueError:
        print('Solution failed to compute')
        intervals = 1

    if method == 'euler':
        for num in range(intervals):
            x0, t0 = euler_step(fun, x0, t0, h, *args)
        if t0 != t1:
            x0, t0 = euler_step(fun, x0, t0, t1 - t0, *args)
    if method == 'runge':
        for num in range(intervals):
            x0, t0 = runge_kutta(fun, x0, t0, h, *args)
        if t0 != t1:
            x0, t0 = runge_kutta(fun, x0, t0, t1 - t0, *args)
    return x0


def solve_ode(x0, t0, t1, eqs, method, deltat_max, *args):
    """
    A function which solves given ODE(s) from initial value (x0, t0) to a given t value using given method and maximum
    step size and returns array of solutions
    :param eqs: ODE(s) to solve
    :param x0: initial x value
    :param t0: initial t value
    :param t1: final t value
    :param deltat_max: maximum step size
    :param method: step type - euler or 4th order runge-kutta
    :param args: additional features/constants to pass into function
    :return: x values at each time value
    """

    if isinstance(x0, tuple) or isinstance(x0, int) or isinstance(x0, np.ndarray):
        pass
    else:
        raise TypeError('Initial conditions wrong type')

    if isinstance(t0, int) or isinstance(t0, float):
        pass
    else:
        raise TypeError('Initial time wrong type')

    if isinstance(t1, int) or isinstance(t1, float):
        pass
    else:
        raise TypeError('Final fime wrong type')

    try:
        is_fun = str(eqs)[1]
        if is_fun == 'f':
            pass
        else:
            raise TypeError('Given ode not a function')
    except IndexError:
        raise TypeError('Given ode not a function')

    if method == 'runge' or method == 'euler':
        pass
    else:
        raise TypeError('Given method not suitable')

    try:
        float(deltat_max)
        pass
    except ValueError:
        raise ValueError('Given delta t wrong type')

    try:
        is_fun = str(eqs)[1]
        if is_fun == 'f':
            try:
                sol1 = eqs(x0, t0, *args)
                if len(sol1) != len(x0):
                    raise IndexError('Unsuitable initial condition dimensions')
            except TypeError:
                raise TypeError('Unsuitable initial conditions dimensions')
        else:
            raise TypeError('Given ode(s) not a function')
    except (IndexError, TypeError):
        raise TypeError('Given ode(s) not a function')

    # if args:
    #     pass
    # else:
    #     try:
    #         ret = eqs(x0, t0)
    #         print(type(ret), 'type')
    #         if isinstance(ret, int) or isinstance(ret, list):
    #             pass
    #         elif isinstance(ret, tuple):
    #             try:
    #                 ret = ret + 1
    #                 print(ret, 'ret')
    #             except TypeError:
    #                 raise TypeError('args needed/initial conditions wrong length')
    #         else:
    #             raise IndexError('args needed/initial conditions wrong length')
    #     except (TypeError, IndexError):
    #         print('args needed/initial conditions wrong length')
    #         return

    x = [x0]
    t_array = []
    t_array = t_array + [t0]
    current_t = t0

    while t1 - current_t > deltat_max:
        current_t += deltat_max
        t_array = t_array + [current_t]
    if current_t != t1:
        t_array = t_array + [t1]

    for no in range(1, len(t_array)):
        if isinstance(x[no-1], list):
            xx = solve_to(eqs, x[no-1][:], t_array[no-1], t_array[no], deltat_max, method, *args)
            x = x + [xx]
        else:
            xx = solve_to(eqs, x[no-1], t_array[no-1], t_array[no], deltat_max, method, *args)
            x = x + [xx]

    return np.array(x)


if __name__ == '__main__':

    # plotting runge, euler and exact solution to dxdt = x to measure method precision

    # Initial Conditions
    x0 = 1
    t0 = 0
    t1 = 1
    deltat_maxx = 0.01

    eul_sol_f = solve_ode(x0, t0, t1, f, 'euler', deltat_maxx)

    run_sol_f = solve_ode(x0, t0, t1, f, 'runge', deltat_maxx)

    t = np.linspace(0, 1, 101)
    plt.plot(t, eul_sol_f)
    plt.plot(t, run_sol_f)
    plt.plot(t, f_true(t))
    plt.show()

    # solving 2 dimensional odes over a long time span to show long term behaviours of respective methods
    # shown that euler spirals outwards while runge spirals inwards slightly, where we expect no diverging
    # or converging at all

    # Initial Conditions f2
    t0 = 0
    t1 = 5
    x0 = 1, 1
    deltat_maxx = 1
    f2_sol = solve_ode(x0, t0, t1, f2, 'runge', deltat_maxx)
    f2_sol_eul = solve_ode(x0, t0, t1, f2, 'euler', deltat_maxx)
    x = []
    xdot = []

    for i in range(0, len(f2_sol)):
        x = x + [f2_sol[i][0]]
        xdot = xdot + [f2_sol[i][1]]

    xe = []
    xdote = []

    for i in range(0, len(f2_sol_eul)):
        xe = xe + [f2_sol_eul[i][0]]
        xdote = xdote + [f2_sol_eul[i][1]]

    plt.plot(x, xdot, label='Runge')
    plt.plot(xe, xdote, label='Euler')
    plt.xlabel('x')
    plt.ylabel('dxdt')
    plt.legend()
    plt.show()

    # error plots for runge and euler methods for dxdt = x

    def error_graph(x, time, time1, fun):

        h_value_list = np.logspace(-4, 0, 50)
        true_x = f_true(time1)

        error_list_eul = np.zeros(int(len(h_value_list)))
        error_list_run = np.zeros(int(len(h_value_list)))

        for i in range(len(h_value_list)):
            eul_sol = solve_ode(x, time, time1, fun, 'euler', h_value_list[i])
            final = eul_sol[-1]
            err = abs(final - true_x)
            error_list_eul[i] = err

        for i in range(len(h_value_list)):
            run_sol = solve_ode(x, time, time1, fun, 'runge', h_value_list[i])
            final = run_sol[-1]
            err = abs(final - true_x)
            error_list_run[i] = err

        ax = plt.gca()
        ax.scatter(h_value_list, error_list_eul, label='Euler')
        ax.scatter(h_value_list, error_list_run, label='RK4')
        ax.set_yscale('log')
        ax.set_xscale('log')
        plt.xlabel('delta t')
        plt.ylabel('Error')
        plt.legend()
        plt.show()

        return error_list_eul, error_list_eul

    x0 = 1
    t0 = 0
    t1 = 1
    eul_err, run_err = error_graph(x0, t0, t1, f)

    # timing euler and runge for similar error

    x0 = 1, 1
    t0 = 0
    t1 = 1

    start_t = time.time()
    solve_ode(x0, t0, t1, f, 'runge', 0.000737)
    final_t_r = time.time() - start_t

    start_t = time.time()
    solve_ode(x0, t0, t1, f, 'euler', 0.4714866)
    final_t_e = time.time() - start_t
    print(final_t_r, final_t_e)
