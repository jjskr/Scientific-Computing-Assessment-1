import numpy as np
import pylab as pl
from math import pi
import scipy.sparse as ssp
from scipy.sparse.linalg import spsolve
import warnings
import time
# bypass efficiency warnings, no real cause for concern
warnings.filterwarnings('ignore', category=ssp.SparseEfficiencyWarning)


def tri_mat(dias, m_len):
    """
    Function returns tridiagonal matrix for chosen method anc boundary condition
    :param dias: for matrix diagonals
    :param m_len: length of matrix
    :return: tridiagonal matrix to use to caculate finite difference
    """
    diags = [[dias[0]] * (m_len - 1), [dias[1]] * m_len, [dias[2]] * (m_len - 1)]
    M = ssp.diags(diags, [-1, 0, 1]).toarray()

    return M


def forward_euler(u_j, lmbda, bc, p, q, deltax, mx, mt, t):
    """
    Function to perform finite differences using forward euler method
    :param u_j: initial conditions
    :param lmbda: lambda value
    :param bc: boundary conditions - dirichlet, neumann, periodic or homogenous
    :param p: LHS boundary condition
    :param q: RHS boundary condition
    :param deltax: change in x
    :param mx: number of x values to solve for
    :param mt: number of intervals between t=0 and T
    :param t: linspace for t
    :return: conditions at t=T
    """
    # diagonals defined
    dias = [lmbda, 1-2*lmbda, lmbda]

    if bc == 'dirichlet':
        # length of matrix required
        m_len = mx-1
        # initiates tridiagonal matrix
        M = tri_mat(dias, m_len)
        # initiates additive vector
        add_v = np.zeros(m_len)

        for j in range(0, mt):
            # for each value in time range, calculation made for new state
            add_v[0], add_v[-1] = p(t[j]), q(t[j])
            add_v_l = add_v * lmbda
            u_jp1 = np.dot(M, u_j[1:-1]) + add_v_l
            u_j[1:-1] = u_jp1
            u_j[0], u_j[-1] = add_v[0], add_v[-1]

    if bc == 'neumann':

        m_len = mx + 1
        M = tri_mat(dias, m_len)
        M[0, 1], M[mx, mx - 1] = 2 * lmbda, 2 * lmbda
        add_v = np.zeros(mx + 1)

        for j in range(0, mt):

            add_v[0], add_v[-1] = -p(t[j]), q(t[j])
            add_v_l = 2 * add_v * lmbda * deltax
            u_j = np.dot(M, u_j) + add_v_l

    if bc == 'periodic':

        M = tri_mat(dias, mx)
        M[0, mx - 1], M[mx - 1, 0] = lmbda, lmbda
        u_j = u_j[:-1]
        for j in range(0, mt):

            u_jp1 = np.dot(M, u_j)
            u_j = u_jp1

    return u_j


def backward_euler(u_j, lmbda, bc, p, q, deltax, mx, mt, t):
    """
    Function to perform finite differences using backward euler method
    :param u_j: initial conditions
    :param lmbda: lambda value
    :param bc: boundary conditions - dirichlet, neumann, periodic or homogenous
    :param p: LHS boundary condition
    :param q: RHS boundary condition
    :param deltax: change in x
    :param mx: number of x values to solve for
    :param mt: number of intervals between t=0 and T
    :param t: linspace for t
    :return: conditions at t=T
    """

    dias = [-lmbda, 1+2*lmbda, -lmbda]

    if bc == 'dirichlet':

        m_len = mx - 1
        M = tri_mat(dias, m_len)
        add_v = np.zeros(mx - 1)

        for j in range(0, mt):

            add_v[0], add_v[-1] = p(t[j]), q(t[j])
            add_v_l = add_v * lmbda
            u_jp1 = spsolve(M, u_j[1:mx]) + add_v_l
            u_j = np.zeros(mx + 1)
            u_j[1:-1] = u_jp1[:]
            u_j[0], u_j[-1] = add_v[0], add_v[-1]

    if bc == 'neumann':

        m_len = mx + 1
        M = tri_mat(dias, m_len)
        M[0, 1], M[mx, mx - 1] = 2 * lmbda, 2 * lmbda
        add_v = np.zeros(mx + 1)

        for j in range(0, mt):

            add_v[0], add_v[-1] = -p(t[j]), q(t[j])
            add_v_l = 2 * add_v * lmbda * deltax
            u_j = spsolve(M, u_j + add_v_l)

    if bc == 'periodic':

        M = tri_mat(dias, mx)
        M[0, mx - 1], M[mx - 1, 0] = lmbda, lmbda

        for j in range(0, mt):

            u_jp1 = spsolve(M, u_j)
            u_j = u_jp1

    return u_j


def crank_nicholson(u_j, lmbda, bc, p, q, deltax, mx, mt, t):
    """
    Function to perform finite differences using crank-nicholson method
    :param u_j: initial conditions
    :param lmbda: lambda value
    :param bc: boundary conditions - dirichlet, neumann, periodic or homogenous
    :param p: LHS boundary condition
    :param q: RHS boundary condition
    :param deltax: change in x
    :param mx: number of x values to solve for
    :param mt: number of intervals between t=0 and T
    :param t: linspace for t
    :return: conditions at t=T
    """

    dia1 = [-lmbda/2, 1+lmbda, -lmbda/2]
    dia2 = [lmbda/2, 1-lmbda, lmbda/2]

    if bc == 'dirichlet':

        m_len = mx - 1
        MA, MB = ssp.csr_matrix(tri_mat(dia1, m_len)), ssp.csr_matrix(tri_mat(dia2, m_len))
        add_v = np.zeros(mx - 1)

        for j in range(0, mt):

            add_v[0], add_v[-1] = p(t[j]), q(t[j])
            add_v_l = add_v * lmbda
            u_jp1 = spsolve(MA, MB*u_j[1:mx]) + add_v_l
            u_j[1:mx] = u_jp1[:]
            u_j[0], u_j[-1] = add_v[0], add_v[-1]

    if bc == 'neumann':

        m_len = mx + 1
        MA, MB = tri_mat(dia1, m_len), tri_mat(dia2, m_len)
        MA[0, 1], MA[mx, mx-1], MB[0, 1], MB[mx, mx-1] = 2 * lmbda, 2 * lmbda, 2 * lmbda, 2 * lmbda
        MA, MB = ssp.csr_matrix(MA), ssp.csr_matrix(MB)
        add_v = np.zeros(mx + 1)

        for j in range(0, mt):

            add_v[0], add_v[-1] = -p(t[j]), q(t[j])
            add_v_l = 2 * add_v * lmbda * deltax
            u_j = spsolve(MA, (MB*u_j)) + add_v_l

            u_j[0] = -add_v[0]
            u_j[-1] = add_v[-1]

    if bc == 'periodic':

        MA, MB = tri_mat(dia1, mx), tri_mat(dia2, mx)
        MA[0, mx - 1], MA[mx - 1, 0], MB[0, mx - 1], MB[mx - 1, 0] = lmbda, lmbda, lmbda, lmbda
        MA, MB = ssp.csr_matrix(MA), ssp.csr_matrix(MB)

        for j in range(0, mt):

            u_jp1 = spsolve(MA, MB*u_j)
            u_j = u_jp1

    return u_j


def solve_pde(mx, mt, method, bc, p, q, plot, args):
    """
    Function initiates the problem and variables, passing them to functions to solve with method desired
    :param mx: number of grid points in space
    :param mt: number of grid points in time
    :param method: desired method to use - FE, BE or CN
    :param bc: boundary type - dirichlet, neumann or periodic
    :param p: left boundary condition
    :param q: right boundary condition
    :param args: additional arguments to pass, kappa, length and time to solve for
    :return: state of pde at time = T
    """

    def u_initial(x):
        # initial temperature distribution
        y = np.sin(pi * x / L)
        return y

    if isinstance(mx, int):
        pass
    else:
        raise TypeError('Number of gridpoints in space wrong type')

    if isinstance(mt, int):
        pass
    else:
        raise TypeError('Number of gridpoints in time wrong type')

    if bc == 'dirichlet':
        pass
    elif bc == 'neumann':
        pass
    elif bc == 'periodic':
        pass
    else:
        raise ValueError('boundary type not suitable')

    if plot == True or plot == False:
        pass
    else:
        raise TypeError('plot condition should be a boolean')

    kappa = args[0]
    L = args[1]
    T = args[2]

    # Set up the numerical environment variables
    x = np.linspace(0, L, mx + 1)  # mesh points in space

    t = np.linspace(0, T, mt + 1)  # mesh points in time
    deltax = x[1] - x[0]  # gridspacing in x
    deltat = t[1] - t[0]  # gridspacing in t
    lmbda = kappa * deltat / (deltax ** 2)  # mesh fourier number

    # Set up the solution variables
    u_j = np.zeros(x.size)  # u at current time step

    for i in range(0, mx + 1):
        u_j[i] = u_initial(x[i])

    if method == 'FE':
        u_j = forward_euler(u_j, lmbda, bc, p, q, deltax, mx, mt, t)
    elif method == 'BE':
        u_j = backward_euler(u_j, lmbda, bc, p, q, deltax, mx, mt, t)
    elif method == 'CN':
        u_j = crank_nicholson(u_j, lmbda, bc, p, q, deltax, mx, mt, t)
    else:
        raise ValueError('method not suitable')

    if plot:
        results_plot(u_j, bc, mx, T)

    return u_j


def results_plot(u_j, bc, mx, T=0.5):

    if bc == 'periodic':
        x = np.linspace(0, L, mx)
    else:
        x = np.linspace(0, L, mx + 1)
    pl.plot(x, u_j, 'ro', label='num')
    xx = np.linspace(0, L, 250)
    pl.plot(xx, u_exact(xx, T), 'b-', label='exact')
    pl.xlabel('x')
    pl.ylabel('u(x,0.5)')
    pl.legend(loc='upper right')
    pl.show()


if __name__ == '__main__':

    def u_exact(x, t):
        # the exact solution
        y = np.exp(-kappa * (pi ** 2 / L ** 2) * t) * np.sin(pi * x / L)
        return y

    def p(t):
        """
        LHS boundary condition
        :param t: time value
        :return: boundary condition at t
        """
        return 0

    def q(t):
        """
        RHS boundary condition
        :param t: time value
        :return: boundary condition at t
        """
        return 0

    # Set problem parameters/functions
    kappa = 1.0  # diffusion constant
    L = 1.0  # length of spatial domain
    T = 0.5  # total time to solve for

    mx = 30  # number of gridpoints in space
    mt = 1000  # number of gridpoints in time
    args = [kappa, L, T]
    u_FE = solve_pde(mx, mt, 'FE', 'dirichlet', p, q, True, args)

    # plot comparing accuracy of each method

    # Set numerical parameters

    # Set problem parameters/functions
    kappa = 1.0  # diffusion constant
    L = 1.0  # length of spatial domain
    T = 0.5  # total time to solve for

    mx = 30  # number of gridpoints in space
    mt = 1000  # number of gridpoints in time
    args = [kappa, L, T]
    for_euler_start = time.time()
    u_FE = solve_pde(mx, mt, 'FE', 'dirichlet', p, q, False, args)
    for_euler_finish = time.time()
    u_BE = solve_pde(mx, mt, 'BE', 'dirichlet', p, q, False, args)
    back_euler_finish = time.time()
    u_CN = solve_pde(mx, mt, 'CN', 'dirichlet', p, q, False, args)
    crank_finish = time.time()
    print('-------------------')
    print('Time Simulations:')
    print('Forward Euler: ', for_euler_finish - for_euler_start)
    print('Backward Euler: ', back_euler_finish - for_euler_finish)
    print('Crank Nicolson: ', crank_finish - back_euler_finish)
    print('-------------------')

    x = np.linspace(0, args[1], mx + 1)
    t = np.linspace(0, args[2], mt + 1)  # mesh points in time
    xx = np.linspace(0, args[1], 250)

    pl.plot(x, u_FE, 'ro', label='FE')
    pl.plot(x, u_BE, 'bo', label='BE')
    pl.plot(x, u_CN, 'ko', label='CN')
    pl.plot(xx, u_exact(xx, args[2]), label='Exact')
    pl.title('Comparing methods with homogenous boundary conditions')
    pl.legend()
    pl.ylabel('u(x,0.5)')
    pl.xlabel('x')
    pl.show()

    # plot showing how lambda changes the accuracy of each method

    mx = 10

    x1 = np.linspace(0, args[1], mx + 1)
    t = np.linspace(0, args[2], mt)  # mesh points in time
    xx = np.linspace(0, args[1], 250)

    g_FE = solve_pde(mx, mt, 'FE', 'dirichlet', p, q, False, args)
    g_BE = solve_pde(mx, mt, 'BE', 'dirichlet', p, q, False, args)
    g_CN = solve_pde(mx, mt, 'CN', 'dirichlet', p, q, False, args)

    fig, (ax1, ax2) = pl.subplots(1, 2, sharey=True)
    ax1.plot(x, u_FE, 'ro', label='FE')
    ax1.plot(x, u_BE, 'bo', label='BE')
    ax1.plot(x, u_CN, 'ko', label='CN')
    ax1.plot(xx, u_exact(xx, T), label='Exact')
    ax1.set_title('lambda = 0.45')
    ax1.legend()
    pl.ylabel('u(x,0.5)')
    pl.xlabel('x')
    ax2.plot(x1, g_FE, 'ro', label='FE')
    ax2.plot(x1, g_BE, 'bo', label='BE')
    ax2.plot(x1, g_CN, 'ko', label='CN')
    ax2.plot(xx, u_exact(xx, T), label='Exact')
    ax2.set_title('lambda = 0.05')
    ax2.legend()
    pl.ylabel('u(x,0.5)')
    pl.xlabel('x')
    pl.show()

    # plotting periodic boundary conditions
    # change boundary conditions need changing accordingly
    # mx = 30
    # p_FE = solve_pde(mx, mt, 'FE', 'periodic', p, q, True, args)
    # x = np.linspace(0, L, mx)
    # xx = np.linspace(0, L, 250)
    # pl.plot(x, p_FE, 'ro', label='FE')
    # pl.plot(xx, u_exact(xx, T))
    # pl.legend()
    # pl.ylabel('u(x,0.5)')
    # pl.xlabel('x')
    # pl.title('Graph showing periodic solution using Forward Euler')
    # pl.show()

    # plot showing how distribution changes with p

    def u_exact_p(x, t, p):
        # the exact solution
        y = np.exp(-kappa * (pi ** 2 / L ** 2) * t) * np.sin(pi * x / L) ** p
        return y

    y_list = []
    labels = []

    x = np.linspace(0, L, 100)  # mesh points in space
    for i in range(1, 10):
        y_list = y_list + [u_exact_p(x, 0.1, i)]
        labels = labels + ['p = ' + str(i)]
    for i in range(len(y_list)):
        pl.plot(x, y_list[i], label=labels[i])
    pl.ylabel('u(x,0.5)')
    pl.xlabel('x')
    pl.legend()
    pl.show()
