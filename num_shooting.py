import numpy as np
import matplotlib.pyplot as plt
import math
from solve_odes import solve_ode


def pp_eqs(X, t):

    x, y = X

    a = 1
    b = 0.26
    d = 0.1

    dxdt = x*(1-x) - a*x*y/(d+x)
    dydt = b*y*(1-(y/x))

    return [dxdt, dydt]


t = np.linspace(0, 100, 1000)
x0 = 0.1, 0.2

sols = solve_ode(x0, t[0], t[len(t)-1], 'euler', 0.1, pp_eqs)

print(sols)

l1 = []
l2 = []

for i in sols:
    l1 = l1 + [i[0]]
    l2 = l2 + [i[1]]
l1 = l1[:-1]
l2 = l2[:-1]

print(l2)
print(t)
plt.plot(l1, t)
plt.plot(l2, t)
plt.show()


