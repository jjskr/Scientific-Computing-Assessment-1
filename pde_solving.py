import numpy as np
import pylab as pl
from math import pi
import scipy.sparse as ssp
from scipy.sparse.linalg import spsolve

# Set problem parameters/functions
kappa = 1.0  # diffusion constant
L = 1.0  # length of spatial domain
T = 0.5  # total time to solve for


def u_initial(x):
    # initial temperature distribution
    y = np.sin(pi * x / L)
    return y


def u_exact(x, t):
    # the exact solution
    y = np.exp(-kappa * (pi ** 2 / L ** 2) * t) * np.sin(pi * x / L)
    return y


def p(t):
    return 0


def q(t):
    return 0


def forward_euler(u_j, lmbda, bc, p, q, deltax):

    if bc == 'dirichlet':

        diag = [[lmbda] * (mx - 2), [1 - 2 * lmbda] * (mx - 1), [lmbda] * (mx - 2)]
        AFE = ssp.diags(diag, [-1, 0, 1]).toarray()
        add_v = np.zeros(mx - 1)

        for j in range(0, mt):
            add_v[0] = p(j)
            add_v[-1] = q(j)
            add_v_l = add_v * lmbda
            u_jp1 = np.dot(AFE, u_j[1:mx] + add_v_l)

            u_j = np.zeros(mx + 1)
            u_j[1:-1] = u_jp1[:]
            u_j[0] = add_v[0]
            u_j[-1] = add_v[-1]

    if bc == 'neumann':

        diag = [[lmbda] * mx, [1 - 2 * lmbda] * (mx + 1), [lmbda] * mx]
        AFE = ssp.diags(diag, [-1, 0, 1]).toarray()
        AFE[0, 1] = 2 * lmbda
        AFE[mx, mx - 1] = 2 * lmbda

        add_v = np.zeros(mx + 1)

        for j in range(0, mt - 1):
            add_v[0] = -p(j)
            add_v[-1] = q(j)

            add_v_l = 2 * add_v * lmbda * deltax

            u_j = np.dot(AFE, u_j + add_v_l)

            u_j[0] = add_v[0]
            u_j[-1] = add_v[-1]
            print(u_j)

    if bc == 'periodic':

        diag = [[lmbda] * (mx - 1), [1 - 2 * lmbda] * mx, [lmbda] * (mx - 1)]
        AFE = ssp.diags(diag, [-1, 0, 1]).toarray()
        AFE[0, mx - 1] = lmbda
        AFE[mx - 1, 0] = lmbda

        for j in range(0, mt):

            u_jp1 = np.dot(AFE, u_j)
            u_j = u_jp1

            u_j[0] = p(j)
            u_j[-1] = q(j)

    return u_j


def backward_euler(u_j, lmbda, bc, p, q, deltax):

    if bc == 'dirichlet':

        diag = [[-lmbda] * (mx - 2), [1 + 2 * lmbda] * (mx - 1), [-lmbda] * (mx - 2)]
        ABE = ssp.diags(diag, [-1, 0, 1]).toarray()
        add_v = np.zeros(mx - 1)

        for j in range(0, mt):
            add_v[0] = p(j)
            add_v[-1] = q(j)
            add_v_l = add_v * lmbda
            u_jp1 = spsolve(ABE, u_j[1:mx] + add_v_l)

            u_j = np.zeros(mx + 1)
            u_j[1:-1] = u_jp1[:]
            u_j[0] = add_v[0]
            u_j[-1] = add_v[-1]

    if bc == 'neumann':

        diag = [[-lmbda] * mx, [1 + 2 * lmbda] * (mx + 1), [-lmbda] * mx]
        ABE = ssp.diags(diag, [-1, 0, 1]).toarray()
        ABE[0, 1] = 2 * -lmbda
        ABE[mx, mx - 1] = 2 * -lmbda
        print(ABE)

        add_v = np.zeros(mx + 1)

        for j in range(0, mt):

            add_v[0] = -p(j)
            add_v[-1] = q(j)

            add_v_l = 2 * add_v * lmbda * deltax

            u_j = spsolve(ABE, u_j + add_v_l)

            u_j[0] = -add_v[0]
            u_j[-1] = add_v[-1]

    if bc == 'periodic':

        diag = [[-lmbda] * (mx-1), [1 + 2 * lmbda] * mx, [-lmbda] * (mx-1)]
        ABE = ssp.diags(diag, [-1, 0, 1]).toarray()
        ABE[0, mx - 1] = lmbda
        ABE[mx - 1, 0] = lmbda

        for j in range(0, mt):

            u_jp1 = spsolve(ABE, u_j)
            u_j = u_jp1

            u_j[0] = p(j)
            u_j[-1] = q(j)

    return u_j


def crank_nicholson(u_j, lmbda, bc, p, q, deltax):

    if bc == 'dirichlet':

        diag_A = [[-lmbda / 2] * (mx - 2), [1 + lmbda] * (mx-1), [-lmbda / 2] * (mx - 2)]
        diag_B = [[lmbda / 2] * (mx - 2), [1 - lmbda] * (mx-1), [lmbda / 2] * (mx - 2)]
        A_CN = ssp.diags(diag_A, [-1, 0, 1])
        B_CN = ssp.diags(diag_B, [-1, 0, 1])
        # diag = [[-lmbda] * (mx - 2), [1 + 2 * lmbda] * (mx - 1), [-lmbda] * (mx - 2)]
        # ABE = ssp.diags(diag, [-1, 0, 1]).toarray()
        add_v = np.zeros(mx - 1)

        for j in range(0, mt):
            add_v[0] = p(j)
            add_v[-1] = q(j)
            add_v_l = add_v * lmbda
            u_jp1 = spsolve(A_CN, B_CN*u_j[1:mx] + add_v_l)

            u_j = np.zeros(mx + 1)
            u_j[1:-1] = u_jp1[:]
            u_j[0] = add_v[0]
            u_j[-1] = add_v[-1]

    if bc == 'neumann':

        diag_A = [[-lmbda / 2] * mx, [1 + lmbda] * (mx+1), [-lmbda / 2] * mx]
        diag_B = [[lmbda / 2] * mx, [1 - lmbda] * (mx+1), [lmbda / 2] * mx]
        A_CN = ssp.diags(diag_A, [-1, 0, 1])
        B_CN = ssp.diags(diag_B, [-1, 0, 1])

        print(A_CN, B_CN)

        add_v = np.zeros(mx + 1)

        for j in range(0, mt):
            add_v[0] = -p(j)
            add_v[-1] = q(j)

            add_v_l = 2 * add_v * lmbda * deltax

            u_j = spsolve(A_CN, B_CN*u_j + add_v_l)

            # u_j = np.zeros(mx + 1)
            # u_j[1:-1] = u_jp1[:]
            u_j[0] = add_v[0]
            u_j[-1] = add_v[-1]

    if bc == 'periodic':

        diag_A = [[-lmbda/2] * (mx-1), [1 + lmbda] * mx, [-lmbda/2] * (mx-1)]
        diag_B = [[lmbda / 2] * (mx-1), [1 - lmbda] * mx, [lmbda / 2] * (mx-1)]

        A_CN = ssp.diags(diag_A, [-1, 0, 1])
        B_CN = ssp.diags(diag_B, [-1, 0, 1])

        for j in range(0, mt):

            u_jp1 = spsolve(A_CN, B_CN*u_j)
            u_j = u_jp1

            u_j[0] = p(j)
            u_j[-1] = q(j)

    return u_j


def solve_pde(mx, mt, method, bc, p, q):

    # Set up the numerical environment variables
    x = np.linspace(0, L, mx + 1)  # mesh points in space
    if bc == 'periodic':
        x = np.linspace(0, L, mx)  # mesh points in space
    t = np.linspace(0, T, mt + 1)  # mesh points in time
    deltax = x[1] - x[0]  # gridspacing in x
    deltat = t[1] - t[0]  # gridspacing in t
    lmbda = kappa * deltat / (deltax ** 2)  # mesh fourier number
    print("deltax=", deltax)
    print("deltat=", deltat)
    print("lambda=", lmbda)

    # Set up the solution variables
    u_j = np.zeros(x.size)  # u at current time step

    # Set initial condition
    if bc == 'periodic':
        for i in range(0, mx):
            u_j[i] = u_initial(x[i])
    else:
        for i in range(0, mx + 1):
            u_j[i] = u_initial(x[i])

    if method == 'FE':
        u_j = forward_euler(u_j, lmbda, bc, p, q, deltax)
    if method == 'BE':
        u_j = backward_euler(u_j, lmbda, bc, p, q, deltax)
    if method == 'CN':
        u_j = crank_nicholson(u_j, lmbda, bc, p, q, deltax)

    results_plot(x, u_j, bc)


def results_plot(x, u_j, bc):
    if bc == 'periodic':
        # Plot the final result and exact solution
        x = np.linspace(0, L, mx)  # mesh points in space
        pl.plot(x, u_j, 'ro', label='num')
        xx = np.linspace(0, L, 250)
        pl.plot(xx, u_exact(xx, T), 'b-', label='exact')
        pl.xlabel('x')
        pl.ylabel('u(x,0.5)')
        pl.legend(loc='upper right')
        pl.show()
    else:
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

solve_pde(mx, mt, 'CN', 'dirichlet', p, q)
