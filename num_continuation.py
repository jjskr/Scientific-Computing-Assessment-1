from scipy.optimize import fsolve
from num_shooting import shooting
import matplotlib.pyplot as plt
from num_shooting import hopf


def cubic_eq(x, args):
    c = args
    cubic_sol = x**3 - x + c
    return cubic_sol


def nat_continuation(f, U0, par_min, par_max, par_step, discretisation):

    par_list = [par_min]
    args = par_min
    cur_par = par_min
    while cur_par + par_step < par_max:
        cur_par += par_step
        par_list = par_list + [cur_par]
    if cur_par != par_max:
        par_list = par_list + [par_max]

    solutions = []

    for par in par_list:
        print(par)
        U0 = fsolve(discretisation(f), U0, args=par)
        solutions = solutions + [U0]

    return par_list, solutions


# cubic initial conditions
U0 = 1.6
pmin = -2
pmax = 2
pstep = 0.01

# par_list, solutions = nat_continuation(cubic_eq, U0, pmin, pmax, pstep, discretisation=lambda x: x)
# plt.plot(par_list, solutions)
# plt.show()
#
# par_list, solutions = nat_continuation(cubic_eq, U0, pmin, pmax, pstep, discretisation=lambda x: x)
# plt.plot(par_list, solutions)
# plt.show()


par_list, solutions = nat_continuation(hopf, U0, pmin, pmax, pstep, shooting)

