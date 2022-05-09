import numpy as np
from solve_odes import f
from solve_odes import f2
from solve_odes import f_true
from solve_odes import solve_ode
# from num_shooting import pp_eqs
import matplotlib.pyplot as plt
from test_for_shooting import hopf_3
from num_shooting import orbit
from num_continuation import nat_continuation
from num_continuation import cubic_eq
from num_continuation import hopfn
from num_shooting import shooting
from num_continuation import hopfm
import math
# from num_shooting import hopf
import time


def error_graph(x, time, time1, fun):

    h_value_list = np.logspace(-4, 0, 50)
    true_x = f_true(time1)

    error_list_eul = np.zeros(int(len(h_value_list)))
    error_list_run = np.zeros(int(len(h_value_list)))

    for i in range(len(h_value_list)):
        eul_sol = solve_ode(x, time, time1, fun, 'euler', h_value_list[i])
        final = eul_sol[-1]
        err = abs(final-true_x)
        error_list_eul[i] = err
        print(err, h_value_list[i])

    for i in range(len(h_value_list)):
        run_sol = solve_ode(x, time, time1, fun, 'runge', h_value_list[i])
        final = run_sol[-1]
        err = abs(final-true_x)
        error_list_run[i] = err
        print(err, h_value_list[i])

    ax = plt.gca()
    ax.scatter(h_value_list, error_list_eul, label='Euler')
    ax.scatter(h_value_list, error_list_run, label= 'RK4')
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.xlabel('delta t')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

    return error_list_eul, error_list_eul


def x_vs_y(fun, x0, t0, t1, deltat_maxx, method='runge', *args):

    f_sol = solve_ode(x0, t0, t1, fun, method, deltat_maxx, *args)
    t_array = []
    t_array = t_array + [t0]
    current_t = t0
    while t1 - current_t > deltat_maxx:
        current_t += deltat_maxx
        t_array = t_array + [current_t]
    if current_t != t1:
        t_array = t_array + [t1]
    plt.plot(t_array, f_sol)
    plt.show()


def x_vs_xdot(x0, t0, t1, deltat_maxx, method='runge'):

    f2_sol = solve_ode(x0, t0, t1, f2, method, deltat_maxx)
    x = []
    xdot = []

    for i in range(0, len(f2_sol)):
        x = x + [f2_sol[i][0]]
        xdot = xdot + [f2_sol[i][1]]

    plt.plot(x, xdot)
    plt.show()


def shooting_plots(U0, ode, *args):

    sol = orbit(ode, U0, args)
    X0, T = sol[:-1], sol[-1]
    sol_mine = solve_ode(X0, 0, T, ode, 'runge', 0.01, *args)
    plt.plot(sol_mine[:, 0], sol_mine[:, 1])
    plt.show()


def shooting_one_cycle(U0, ode, *args):

    sol = orbit(ode, U0, args)
    X0, T = sol[:-1], sol[-1]
    sol_mine = solve_ode(X0, 0, T, ode, 'runge', 0.01, *args)
    time_cycle = math.ceil(float(T)/0.01) + 1
    t = np.linspace(0, T, time_cycle)
    plt.plot(t, sol_mine)
    plt.show()


# # Week 1
#
# # # error graph showing difference between RK4 and Euler methods
#
x0 = 1
t0 = 0
t1 = 1
error_graph(x0, t0, t1, f)

# from error list, found timesteps for each that result in similar errors, used time to measure time they take to
# compute
start_t = time.time()
solve_ode(x0, t0, t1, f, 'runge', 0.000737)
final_t_r = time.time() - start_t

start_t = time.time()
solve_ode(x0, t0, t1, f, 'euler', 0.4714866)
final_t_e = time.time() - start_t
print('runge-kutta: ' + str(final_t_r), 'euler: ' + str(final_t_e))
#
# # graph showing x and y vs time
#
# t0 = 0
# t1 = 10
# x0 = 1, 1
# deltat_maxx = 0.01
#
# x_vs_y(f2, x0, t0, t1, deltat_maxx)
#
# # graph showing x vs x dot
#
# t0 = 0
# t1 = 100
# x0 = 1, 1
# deltat_maxx = 1
# x_vs_xdot(x0, t0, t1, deltat_maxx)

# # FIGURES FOR SHOOTING
#
# # b < 0.26
#
# X0 = 0.2, 0.3
# T = 21
# U0 = 0.2, 0.3, 21
# args = [1, 0.2, 0.1]
# ode = pp_eqs
# x_vs_y(ode, X0, 0, 300, 0.1, 'runge', args)
#
# # b = 0.26
#
# args[1] = 0.26
# x_vs_y(ode, X0, 0, 300, 0.1, 'runge', args)
#
# # b > 0.26
#
# args[1] = 0.3
# x_vs_y(ode, X0, 0, 300, 0.1, 'runge', args)

# # hopf 3 graph
# args = [1, -1]
# U0 = 1, 1, 1, 6
# ode = hopf_3
# U0 = orbit(ode, U0, args)
# X0, T = U0[:-1], U0[-1]
# x_vs_y(ode, X0, 0, 20, 0.01, 'runge', args)


# pp shooting
# # initial guess predator-prey
# method = 'runge'
# X0 = 0.2, 0.3
# T = 21
# U0 = 0.2, 0.3, 21
# args = [1, 0.1, 0.1]
# ode = pp_eqs
# shooting_plots(U0, ode, args)
# shooting_one_cycle(U0, ode, args)


# hopf shooting
# initial guess hopf
# method = 'runge'
# ode = hopf
# args = [1, -1]
# U0 = 1.5, 1.5, 5
# X0 = 1.5, 1.5
# T = 5

# shooting_plots(U0, ode, args)
# shooting_one_cycle(U0, ode, args)


# hopf3
# initial guess hopf 3 odes

ode = hopf_3
U0 = 1, 1, 1, 8
X0 = 1, 1, 1
T = 8
args = [1, -1]

# shooting_plots(U0, ode, args)
# shooting_one_cycle(U0, ode, args)


# NUMERICAL CONTINUATION

# # cubic initial conditions
# U0 = 1.6
# pmin = -2
# pmax = 2
# pstep = 100
#
# par_list, solutions = nat_continuation(cubic_eq, U0, pmin, pmax, pstep, discretisation=lambda x: x)
# plt.plot(par_list, solutions)
# plt.show()

# # hopfn
#
# U0 = 1.4, 0, 6.3
# pmin = 2
# pmax = 0
# pstep = 50
#
# par_list, solutions = nat_continuation(hopfn, U0, pmin, pmax, pstep, shooting)
# plt.plot(par_list, solutions)
# plt.show()

# # hopfm
#
# U0 = 1.4, 0, 6.3
# pmin = 2
# pmax = -1
# pstep = 34
# pstep = 40
#
# par_list, solutions = nat_continuation(hopfm, U0, pmin, pmax, pstep, shooting)
# plt.plot(par_list, solutions)
# plt.show()
