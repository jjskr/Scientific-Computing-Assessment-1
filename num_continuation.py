from scipy.optimize import fsolve
from num_shooting import shooting
import matplotlib.pyplot as plt
from num_shooting import hopf
import numpy as np


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


def nat_continuation(f, U0, par_min, par_max, par_split, discretisation):

    params = np.linspace(par_min, par_max, par_split)

    solutions = []

    for par in params:
        print(U0, par)
        U0 = fsolve(discretisation(f), U0, par)
        U0 = np.round(U0, 5)
        solutions = solutions + [U0[0]]

    return params, solutions



# cubic initial conditions
U0 = 1.6
pmin = -2
pmax = 2
pstep = 100

# par_list, solutions = nat_continuation(cubic_eq, U0, pmin, pmax, pstep, discretisation=lambda x: x)
# plt.plot(par_list, solutions)
# plt.show()

U0 = 1, 1, 6
pmin = 2
pmax = 0
pstep = 30

# par_list, solutions = nat_continuation(hopfn, U0, pmin, pmax, pstep, shooting)
# plt.plot(par_list, solutions)
# plt.show()

pmin = 2
pmax = -1
par_list, solutions = nat_continuation(hopfm, U0, pmin, pmax, pstep, shooting)

plt.plot(par_list, solutions)
plt.show()

# U0 = 1.4, 0, 6.3
# params = [2, -1]
# pmin = -1
# pmax = 2
# pstep = 0.1
#
# par_list, solutions = nat_continuation_h(hopf, U0, params, pmin, pmax, pstep, 0, shooting)

