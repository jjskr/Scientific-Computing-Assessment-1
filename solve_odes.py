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

    k1 = f(x, t)
    k2 = f((x + delta_t * k1/2), (t + delta_t/2))
    k3 = f((x + delta_t * k2/2), (t + delta_t/2))
    k4 = f((x + delta_t * k3), (t + delta_t))
    k = (k1+2*k2+2*k3+k4)/6
    t = t + delta_t
    x = x + delta_t * k
    return x, t


def solve_to(x0, t0, t1, h, step, intervals, fun):
    #
    # intervals = (t2-t)/h
    #
    # if isinstance(intervals, int):
    #     delta_t = h
    # else:
    #     intervals = math.ceil(intervals)
    #     delta_t = (t2-t)/intervals

    intervals = round((t1-t0)/h, 5)
    # print(intervals)

    if isinstance(intervals, int):
        delta_t = h
    else:
        intervals = math.ceil(intervals)
        delta_t = (t1-t0)/intervals

    xn = x0
    tn = t0

    if step == 'euler':
        for num in range(intervals):
            xn, tn = euler_step(fun, xn, tn, delta_t)
            # print(tn)
    elif step == 'runge':
        for num in range(intervals):
            xn, tn = runge_kutta(fun, xn, tn, delta_t)
            # print(xn)
            # print(xn)
    return xn


def solve_ode(x0, t0, t1, step, deltat_max, eqs):

    x = [np.array(x0)]
    t_array = []

    intervals = (t1-t0)/deltat_max

    if isinstance(intervals, int):
        delta_t = deltat_max
    else:
        intervals = math.ceil(intervals)
        delta_t = (t1-t0)/intervals

    while round(t0, 8) < t1:
        t_array = t_array + [t0]
        t0 += delta_t
    t_array = t_array + [t1]

    fun = eqs

    for no in range(1, len(t_array)):
        # print(no)
        # print(t_array[no])
        if isinstance(x[no-1], list):
            xx = solve_to(x[no-1][:], t_array[no-1], t_array[no], delta_t, step, intervals, fun)
            x = x + [xx]
        else:
            xx = solve_to(x[no-1], t_array[no-1], t_array[no], delta_t, step, intervals, fun)
            x = x + [xx]

    # Plots system of odes
    # l1 = []
    # l2 = []
    # for i in x:
    #     l1 = l1 + [i[0]]
    #     l2 = l2 + [i[1]]
    # plt.plot(l1, t_array)
    # plt.plot(l2, t_array)
    # plt.show()

    return x

# def solve_ode(x, t, t2, h):
#
#     delta_t = solve_to(t, t2, h)
#     array_x = [x]
#     array_t = [t]
#     # print('Enter method:')
#     method = 'euler'
#     if method == 'euler':
#         while t < t1:
#             x, t = euler_step(x, t, delta_t)
#             array_x = array_x + [x]
#             array_t = array_t + [round(t, 5)]
#             error = abs(array_x[len(array_x)-1] - f_true(x, t))
#
#     elif method == 'runge kutta':
#         while t < t1:
#             x, t = runge_kutta(x, t, delta_t)
#             array_x = array_x + [x]
#             array_t = array_t + [round(t, 5)]
#     return array_x, array_t, error


def error_graph(x, time, time1):

    h_value_list = np.logspace(-4, -1, 50)
    # print(h_value_list)
    true_x = f_true(x, time1)

    error_list_eul = np.zeros(int(len(h_value_list)))
    error_list_run = np.zeros(int(len(h_value_list)))

    # for i in range(len(h_values)):
    #     xxx = solve_ode1(x, time, time1, 'euler', h_value_list[i])
    #     final = xxx[-1]
    #     err = abs(final-true_x)
    #     error_list_eul[i+1] = err

    for i in range(len(h_values)):
        ppp = solve_ode(x, time, time1, 'runge', h_value_list[i], fun)
        final = ppp[-1]
        err = abs(final-true_x)
        error_list_run[i+1] = err

    print(error_list_run)
    # print(len(error_list_run))

    ax = plt.gca()
    ax.scatter(h_value_list, error_list_eul)
    ax.scatter(h_value_list, error_list_run)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.margins(2, 2)
    plt.show()


# Initial Conditions
x0 = 0, 2
t0 = 0
t1 = 50
deltat_max = 0.1

h_values = np.linspace(t0, t1, 100)
# print(len(h_values))
h_values = h_values[1:50]

error_values = []
delta_t_values = []

actual = np.exp(t1)
# print(h_values)

# for i in range(0, len(h_values)-1):
#     array_x, array_t, error1 = solve_ode(x0, t0, t1, h_values[i])
#     error1 = abs(array_x[len(array_x)-1] - actual)
#     # print(array_x[len(array_x)-1])
#     if error1 not in error_values:
#         error_values = error_values + [error1]
#         delta_t_values = delta_t_values + [solve_to(t0, t1, h_values[i])]

xxx = solve_ode(x0, t0, t1, 'euler', deltat_max, f2)
print(xxx)

# print((error_values))
# print((h_values))

# t = np.linspace(t0, t1, 1000)
# y = np.exp(t)
# plt.plot(array_t, array_x)
# plt.plot(t, y)
# plt.xlabel('$t$')
# plt.ylabel('$x$')
# plt.show()
# plt.scatter(delta_t_values, error_values)
# plt.show()

# error_graph(x0, t0, t1)
