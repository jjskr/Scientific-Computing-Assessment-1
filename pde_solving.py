import numpy as np
import pylab as pl
from math import pi
import scipy.sparse as ssp
from scipy.sparse.linalg import spsolve

# Set problem parameters/functions
kappa = 1.0  # diffusion constant
L = 1.0  # length of spatial domain
T = 0.5  # total time to solve for


def u_I(x):
    # initial temperature distribution
    y = np.sin(pi * x / L)
    # y = np.sin(pi * x)**6
    return y


def u_exact(x, t):
    # the exact solution
    y = np.exp(-kappa * (pi ** 2 / L ** 2) * t) * np.sin(pi * x / L)
    return y


def forward_euler(u_j, u_jp1, lmbda):

    # initialise matrix
    diag = [[lmbda] * (mx - 2), [1 - 2 * lmbda] * (mx-1), [lmbda] * (mx - 2)]
    AFE = ssp.diags(diag, [-1, 0, 1]).toarray()

    for j in range(0, mt):
        u_jp1[1:mx] = np.dot(AFE, u_j[1:mx])

        # Boundary conditions
        u_jp1[0] = 0
        u_jp1[mx] = 0

        # Save u_j at time t[j+1]
        u_j[:] = u_jp1[:]

    return u_j


def backward_euler(u_j, u_jp1, lmbda):

    # initialise matrix
    diag = [[-lmbda] * (mx - 2), [1 + 2 * lmbda] * (mx-1), [-lmbda] * (mx - 2)]
    ABE = ssp.diags(diag, [-1, 0, 1]).toarray()

    for j in range(0, mt):
        u_jp1[1:mx] = spsolve(ABE, u_j[1:mx])

        # Boundary conditions
        u_jp1[0] = 0
        u_jp1[mx] = 0

        # Save u_j at time t[j+1]
        u_j[:] = u_jp1[:]

    return u_j


def crank_nicholson(u_j, u_jp1, lmbda):

    diag_A = [[-lmbda/2] * (mx - 1), [1 + lmbda] * mx, [-lmbda/2] * (mx - 1)]
    diag_B = [[lmbda/2] * (mx - 1), [1 - lmbda] * mx, [lmbda/2] * (mx - 1)]
    A_CN = ssp.diags(diag_A, [-1, 0, 1])
    B_CN = ssp.diags(diag_B, [-1, 0, 1])

    for j in range(0, mt):
        u_jp1[1:] = spsolve(A_CN, B_CN*u_j[1:])

        # Boundary conditions
        u_jp1[0] = 0
        u_jp1[mx] = 0

        # Save u_j at time t[j+1]
        u_j[:] = u_jp1[:]

    return u_j


def solve_pde(mx, mt, method):

    # Set up the numerical environment variables
    x = np.linspace(0, L, mx + 1)  # mesh points in space
    t = np.linspace(0, T, mt + 1)  # mesh points in time
    deltax = x[1] - x[0]  # gridspacing in x
    deltat = t[1] - t[0]  # gridspacing in t
    lmbda = kappa * deltat / (deltax ** 2)  # mesh fourier number
    print("deltax=", deltax)
    print("deltat=", deltat)
    print("lambda=", lmbda)

    # Set up the solution variables
    u_j = np.zeros(x.size)  # u at current time step
    u_jp1 = np.zeros(x.size)  # u at next time step

    # Set initial condition
    for i in range(0, mx + 1):
        u_j[i] = u_I(x[i])

    if method == 'FE':
        u_j = forward_euler(u_j, u_jp1, lmbda)
    if method == 'BE':
        u_j = backward_euler(u_j, u_jp1, lmbda)
    if method == 'CN':
        u_j = crank_nicholson(u_j, u_jp1, lmbda)

    # # Solve the PDE: loop over all time points
    # for j in range(0, mt):
    #     # Forward Euler timestep at inner mesh points
    #     # PDE discretised at position x[i], time t[j]
    #     for i in range(1, mx):
    #         u_jp1[i] = u_j[i] + lmbda * (u_j[i - 1] - 2 * u_j[i] + u_j[i + 1])
    #

    results_plot(x, u_j)


def results_plot(x, u_j):

    # Plot the final result and exact solution
    pl.plot(x, u_j, 'ro', label='num')
    xx = np.linspace(0, L, 250)
    pl.plot(xx, u_exact(xx, T), 'b-', label='exact')
    pl.xlabel('x')
    pl.ylabel('u(x,0.5)')
    pl.legend(loc='upper right')
    pl.show()


# Set numerical parameters
mx = 10  # number of gridpoints in space
mt = 1000  # number of gridpoints in time

solve_pde(mx, mt, 'CN')



