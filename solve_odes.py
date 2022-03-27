import numpy as np
import matplotlib.pyplot as plt
import math


def f(x, t):
    return x


def f2(X, t):
    x, y = X
    dxdt = y
    dydt = -x
    return [dxdt, dydt]


def f_true(x, t):
    x = np.exp(t)
    return x


def euler_step(f, x, t, delta_t):

    x = x + delta_t * np.array(f(x, t))
    t = t + delta_t

    return x, t


def runge_kutta(f, x, t, delta_t):

    k1 = np.array(f(x, t))
    k2 = np.array(f((x + delta_t * k1/2), (t + delta_t/2)))
    k3 = np.array(f((x + delta_t * k2/2), (t + delta_t/2)))
    k4 = np.array(f((x + delta_t * k3), (t + delta_t)))
    k = (k1+2*k2+2*k3+k4)/6
    t = t + delta_t
    x = x + delta_t * k

    return x, t


def solve_to(fun, x0, t0, t1, h, method='runge'):

    t_diff = t1 - t0
    # print(t_diff)
    intervals = math.floor(t_diff/h)

    if method == 'euler':
        for num in range(intervals):
            x0, t0 = euler_step(fun, x0, t0, h)
        if t0 != t1:
            x0, t0 = euler_step(fun, x0, t0, t1 - t0)
    if method == 'runge':
        for num in range(intervals):
            x0, t0 = runge_kutta(fun, x0, t0, h)
        if t0 != t1:
            x0, t0 = runge_kutta(fun, x0, t0, t1 - t0)
    return x0


def solve_ode(x0, t0, t1, eqs, method='runge', deltat_max=0.01):

    x = [np.array(x0)]
    t_diff = t1-t0
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
            xx = solve_to(eqs, x[no-1][:], t_array[no-1], t_array[no], deltat_max, method)
            x = x + [xx]
        else:
            xx = solve_to(eqs, x[no-1], t_array[no-1], t_array[no], deltat_max, method)
            x = x + [xx]

    return np.array(x)


def error_graph(x, time, time1, fun):

    h_value_list = np.logspace(-4, -1, 50)
    true_x = f_true(x, time1)

    error_list_eul = np.zeros(int(len(h_value_list)))
    error_list_run = np.zeros(int(len(h_value_list)))

    for i in range(len(h_value_list)):
        eul_sol = solve_ode(x, time, time1, fun, 'euler', h_value_list[i])
        final = eul_sol[-1]
        err = abs(final-true_x)
        error_list_eul[i] = err

    for i in range(len(h_value_list)):
        run_sol = solve_ode(x, time, time1, fun, 'runge', h_value_list[i])
        final = run_sol[-1]
        err = abs(final-true_x)
        error_list_run[i] = err

    # print(len(error_list_run))
    # print(len(error_list_eul))
    # print(error_list_eul)
    # print(error_list_run)

    ax = plt.gca()
    ax.scatter(h_value_list, error_list_eul)
    ax.scatter(h_value_list, error_list_run)
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.show()


if __name__=='__main__':

    # Initial Conditions
    x0 = 1
    t0 = 0
    t1 = 1
    deltat_maxx = 0.01

    error_values = []
    delta_t_values = []

    actual = np.exp(t1)

    xxx = solve_ode(x0, t0, t1, f, 'euler', deltat_maxx)

    error_graph(x0, t0, t1, f)
