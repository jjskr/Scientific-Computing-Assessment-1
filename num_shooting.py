import numpy as np
import matplotlib.pyplot as plt
import math
from solve_odes import solve_ode
from scipy.optimize import fsolve


def shooting(ode):
    """
    Uses chosen ODE to set function to solve to find orbit cycle of systems of ODEs
    :param ode: System of odes
    :return: Conditions function
    """
    def conditions(U0, pc, *args):
        """
        Function needed to be satisfied to calculate limit cycle
        :param U0: current conditions
        :param pc: Phase condition to use
        :param args: additional arguments
        :return: conditions to be satisfied for calculating a limit cycle
        """
        # set up initial conditions to pass into solve_ode
        X0, T = U0[:-1], U0[-1]
        # find runge kutta solutions at T
        sols = solve_ode(X0, 0, T, ode, 'runge', 0.01, *args)
        sol = sols[-1, :]
        # find current phase condition
        phase_cond = pc(X0, T, ode, *args)
        # find current period condition
        period_cond = []
        for i in range(len(sol)):
            period_cond = period_cond + [X0[i] - sol[i]]

        return np.r_[period_cond, phase_cond]

    return conditions


def orbit(ode, U0, pc, *args):
    """
    Calculates solution to shooting(ode) using fsolve
    :param ode: System of odes to solve for
    :param U0: Initial conditions
    :param pc: Phase condition to use
    :param args: Arguments needed by system of odes
    :return: Solution of initial conditions and time period of one orbit in the form of an array
    """

    # simple input tests
    try:
        ini_cons = len(U0)
        if ini_cons < 2:
            raise IndexError('Initial conditions too short')
    except TypeError:
        raise TypeError('Initial conditions wrong type')

    try:
        is_fun = str(ode)[1]
        if is_fun == 'f':
            try:
                sol1 = ode(U0[:-1], U0[-1], *args)
                if len(sol1) != len(U0[:-1]):
                    raise IndexError('Unsuitable initial condition dimensions')
            except (TypeError, ValueError):
                raise TypeError('Unsuitable initial conditions dimensions')
        else:
            raise TypeError('Given ode(s) not a function')
    except (IndexError, TypeError):
        raise TypeError('Given ode(s) not a function')

    try:
        is_fun = str(pc)[1]
        if is_fun == 'f':
            pass
        else:
            raise TypeError('Given phase condition not a function')
    except (IndexError, TypeError):
        raise TypeError('Given phase condition not a function')

    # uses fsolve to find root of shooting function
    sol = fsolve(shooting(ode), U0, (pc, *args))

    return sol


if __name__ == '__main__':

    def pp_eqs(U0, t, args):
        """
        A function that returns the values of the given predator prey functions at (U0, t)
        :param U0: values for x and y
        :param t: time
        :param args: list of a, b and d constants
        :return: solutions to predator prey equations
        """

        x, y = U0

        a = args[0]
        b = args[1]
        d = args[2]

        dxdt = x * (1 - x) - a * x * y / (d + x)
        dydt = b * y * (1 - (y / x))

        return [dxdt, dydt]


    def hopf(U0, t, args):
        """
        A function that returns solutions to the hopf equations at (U0, t)
        :param U0: values for x and y
        :param t: time
        :param args: list of beta and sigma constants
        :return: solutions to hopf equations
        """
        # print(U0, t)
        beta = args[0]
        sigma = args[1]
        u1, u2 = U0
        du1dt = beta * u1 - u2 + sigma * u1 * (u1 ** 2 + u2 ** 2)
        du2dt = u1 + beta * u2 + sigma * u2 * (u1 ** 2 + u2 ** 2)

        return [du1dt, du2dt]


    def pc_stable_0(U0, T, ode, args):
        return ode(U0, 0, args)[0]


    def shooting_plots(U0, ode, pc, args):
        """
        Function to plot orbit shooting solutions
        :param U0: initial conditions
        :param ode: ODE(s) to solve for
        :param pc: Phase condition to use
        :param args: additional arguments for function
        :return: plots of orbit birufication
        """

        sol = orbit(ode, U0, pc, args)
        X0, T = sol[:-1], sol[-1]
        mysol = solve_ode(X0, 0, T, ode, 'runge', 0.01, args)
        plt.plot(mysol[:, 0], mysol[:, 1])
        plt.title('Phase portrait for given ode')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()


    def shooting_one_cycle(U0, ode, pc, args):
        """
        Function returning plots of x and y against time
        :param U0: initial conditions
        :param ode: ODE(s) to solve for
        :param pc: Phase condition to use
        :param args: additional arguments to pass onto function
        :return: plots of cycle vs time
        """

        sol = orbit(ode, U0, pc, args)
        X0, T = sol[:-1], sol[-1]
        mysol = solve_ode(X0, 0, T, ode, 'runge', 0.01, args)
        time_cycle = math.ceil(float(T) / 0.01) + 1
        t = np.linspace(0, T, time_cycle)
        plt.plot(t, mysol)
        plt.xlabel('Time (s)')
        plt.legend(['x', 'y'])
        plt.title('ODE for one limit cycle in time domain')
        plt.show()

    # plotting predator prey equations with different b value to assess the systems behaviour

    # initial guess predator-prey
    X0 = 0.2, 0.3
    T = 21
    U0 = 1, 1, 20
    args = [1, 0.26, 0.1]
    ode = pp_eqs
    pc = pc_stable_0

    # b > 0.26 - graph shows that function converges
    args[1] = 0.3
    sol_mine_less = solve_ode(X0, 0, 100, ode, 'runge', 0.01, args)
    t_val = math.ceil(100/0.01) + 1
    t = np.linspace(0, 100, t_val)
    plt.plot(t, sol_mine_less)
    plt.title('Graph showing how predator prey equations vary over time, b > 0.26')
    plt.show()

    # b < 0.26 - graph shows that function diverges
    args[1] = 0.16
    sol_mine_more = solve_ode(X0, 0, 100, ode, 'runge', 0.01, args)
    t_val = math.ceil(100/0.01) + 1
    t = np.linspace(0, 100, t_val)
    plt.plot(t, sol_mine_more)
    plt.title('Graph showing how predator prey equations vary over time, b < 0.26')
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(t, sol_mine_less)
    ax1.set_title('beta < 0.26')
    ax2.plot(t, sol_mine_more)
    ax2.set_title('beta > 0.26')
    plt.show()

    # figure showing how phase portrait varies with beta
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(sol_mine_less[:, 0], sol_mine_less[:, 1])
    ax1.set_title('beta < 0.26')
    ax2.plot(sol_mine_more[:, 0], sol_mine_more[:, 1])
    ax2.set_title('beta > 0.26')
    plt.show()

    # finding orbit manually
    # isolated by inspection - ran in great detail to ensure the period was found correctly
    args = [1, 0.26, 0.1]
    sol_mine = solve_ode(X0, 0, 100, ode, 'runge', 0.01, args)
    t_val = math.ceil(100/0.01) + 1
    t = np.linspace(0, 100, t_val)
    plt.plot(t, sol_mine)
    plt.show()

    # using conditions and time from orbit function to plot both phase portraits and time series
    # of predator prey equations
    # plotting phase portrait
    shooting_plots(U0, ode, pc, args)

    # plotting orbit found by orbit function for predator prey equations
    shooting_one_cycle(U0, ode, pc, args)
