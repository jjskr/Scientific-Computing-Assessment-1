import numpy as np
import matplotlib.pyplot as plt
import math
from solve_odes import solve_ode



def pp_eqs(X, t):

    x, y = X

    a = 1
    b = 0.2
    d = 0.1

    dxdt = x*(1-x) - a*x*y/(d+x)
    dydt = b*y*(1-(y/x))

    return [dxdt, dydt]



