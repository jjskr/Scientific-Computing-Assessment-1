import numpy as np
import matplotlib.pyplot as plt
import math
from solve_odes import solve_ode
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp


def pp_eqs_ivp(t, X):
    x, y = X

    a = 1
    b = 0.15
    d = 0.1

    dxdt = x * (1 - x) - a * x * y / (d + x)
    dydt = b * y * (1 - (y / x))

    return [dxdt, dydt]


def pp_eqs(X, t):
    x, y = X

    a = 1
    b = 0.16
    d = 0.1

    dxdt = x*(1-x) - a*x*y/(d+x)
    dydt = b*y*(1-(y/x))

    return [dxdt, dydt]


# def conds(U0):
#     X0, T = U0[:-1], U0[-1]
#     print('initial: ', X0)
#     print('T: ', T)
#     sol = solve_ivp(pp_eqs_ivp, (0, T), X0).y
#     pc_sol = sol[:, -1]
#     phase_cond = pp_eqs_ivp(0, X0)[0]
#     period_cond = X0 - pc_sol
#     print(X0, pc_sol)
#     print(period_cond, T)
#
#     return np.r_[phase_cond, period_cond]


def conds(U0):
    X0, T = U0[:-1], U0[-1]
    sols = solve_ode(X0, 0, T, pp_eqs)
    sol = sols[-1, :]
    phase_cond = pp_eqs(X0, 0)[0]
    period_cond = [X0[0] - sol[0], X0[1] - sol[1]]
    print(X0, sol, T)

    # print('sol: ', sol[-1, :])

    return np.r_[phase_cond, period_cond]


def shooting(X0, T):
    sol = fsolve(conds, np.r_[X0, T])
    print(T)
    return sol


X0 = 0.3, 0.3
# X0 = 0.27015621,  0.27015621
T = 20.39950159
# T = 15.523973063297696
t_span = np.linspace(0, T, 200)
# temp_sol = solve_ivp(lambda t,X:pp_eqs(X,t),(0,T),X0,t_eval=t_span).y
# plt.plot(temp_sol.T)
# plt.show()
# quit()

# t = np.linspace(0, 200, 1001)
X0 = 0.2, 0.4
T = 21

sol = shooting(X0, T)
# print('sol: ',sol)
X0, T = sol[:-1], sol[-1]
print('T', T)
print('X0: ', X0)
t_span = np.linspace(0, T, 200)
# sol = solve_ivp(pp_eqs_ivp, (0, T), X0, t_eval=t_span).y.T
sol_mine = solve_ode(X0, 0, T, pp_eqs)
# print('solve ode: ', sol_mine)
# print('solve ivp: ', sol)
plt.plot(sol_mine)
# plt.figure()
# plt.plot(sol[:, 0], sol[:, 1])
plt.show()
# delta_t = t[1]-t[0]

# sols = solve_ode(x0, t[0], t[len(t)-1], 'runge', delta_t, pp_eqs)

# print(t)
# print(phase_con(x0, t))

# l1 = []
# l2 = []
#
# for i in sols:
#     l1 = l1 + [i[0]]
#     l2 = l2 + [i[1]]
# l1 = l1[:-1]
# l2 = l2[:-1]


# print(l1)
# plt.plot(t, l1)
# plt.plot(t, l2)
# plt.show()
#
# plt.plot(l1,l2)
# plt.show()

