"""
Written by
    Nathanael J. Reynolds
        SDSU, 2021
            Version 3.0
"""

# ===================================================
# Version notes
# ===================================================
# Version 1.1
# Syntax for Adams Bashforth, AB/AM, and fourth order Adams Moulton
# updated to be more compact
#
# Version 2.0
# Euler-Mayurama added to calculate stochastic ODEs
#
# Version 3.0
# Dynamical systems: lorenz, rossler, chua, and signum added to test
# the schemes

import numpy as np
import matplotlib.pyplot as plt


class Scheme:
    """
    A collection of finite difference schemes to numerically intergrate ordinary differential equations
    """

    def __init__(self,
                 differential_equation,
                 initial_condition,
                 start_time,
                 stop_time,
                 time_step):
        """
        :param differential_equation: mathematical function representing the ODE may be single equation or Nx1
        			      array of equations: e.g. np.array([[element1], [element2], ..., [elementN]])
	    :param initial_condition:     initial conditions of the ODE. Must be expressed as a Nx1
        			      array: e.g. np.array([[element1], [element2], ..., [elementN]])			  
        :param start_time: 	      start time of the measurement expressed as an integer
        :param stop_time: 	      stop time of the measurement expressed as an integer
        :param time_step: 	      time step/resolution of the scheme's grid expressed as an integer
        			      appears as h or dt in analytic work
        """
        self.f = differential_equation
        self.size = initial_condition[:].size
        self.y0 = initial_condition
        self.h = time_step
        self.t = Grid(start_time, stop_time, time_step).create_grid()

    def abam(self):
        """
        :return: numerical approximation of ODE solution using the Adams-Moulton Scheme
        	 and name of the scheme
        """
        I = np.size(self.t)
        y = np.zeros((self.size, I))
        initial_step, name = Scheme.runge_kutta(self)
        for i in range(0, I):
            if i == 0:
                y[:, i] = self.y0[:]
            elif i < 4: 
                y[:, i] = initial_step[:, i]
            else: 
                y[:, i] = y[:, i-1] + self.h/24 * (55*self.f(self.t[i-1], y[:, i-1]) -
                                                   59*self.f(self.t[i-2], y[:, i-2]) +
                                                   37*self.f(self.t[i-3], y[:, i-3]) -
                                                    9*self.f(self.t[i-4], y[:, i-4]))
                y[:, i] = y[:, i-1] + self.h/24 * (9*self.f(self.t[i], y[:, i]) +
                                                  19*self.f(self.t[i-1], y[:, i-1]) -
                                                   5*self.f(self.t[i-2], y[:, i-2]) +
                                                     self.f(self.t[i-3], y[:, i-3]))
        return y, 'AB/AM predictor'



    def adams_bashforth(self):
        """
        :return: numerical approximation of ODE solution using the Second Order Adams-Bashforth Scheme
        	 and name of the scheme
        """
        I = np.size(self.t)
        y = np.zeros((self.size, I))
        initial_step, name = Scheme.runge_kutta(self)
        for i in range(0, I):
            if i == 0:
                y[:, i] = self.y0[:]
            elif i == 1:
                y[:, i] = initial_step[:, i]
            else:
                y[:, i] = y[:, i-1] + 3*self.h/2 * self.f(self.t[i-1], y[:, i-1]) -\
                          self.h/2 * self.f(self.t[i-2], y[:, i-2])
        return y, 'adams bashforth'

    def adams_moulton(self):
        """
        :return: numerical approximation of ODE solution using the Second-Order Adams-Moulton Scheme
         	 and name of the scheme
        """
        I = np.size(self.t)
        y = np.zeros((self.size, I))
        one = np.ones((self.size, I))
        for i in range(0, I):
            if i == 0:
                y[:, i] = self.y0[:]
            else:
                k1 = self.f(self.t[i - 1], y[:, i - 1])
                k2 = self.f(self.t[i - 1] + self.h / 2, y[:, i - 1] + k1 * one[:, i] * self.h / 2)
                k3 = self.f(self.t[i - 1] + self.h / 2, y[:, i - 1] + k2 * one[:, i] * self.h / 2)
                k4 = self.f(self.t[i - 1] + self.h, y[:, i - 1] + k2 * one[:, i] * self.h)
                y[:, i] = y[:, i - 1] + self.h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                y[:, i] = y[:, i-1] + self.h/2 * (self.f(self.t[i-1], y[:, i-1]) +
                                                  self.f(self.t[i], y[:, i]))
        return y, 'adams moulton'

    def adams_moulton4(self):
        """
        :return: numerical approximation of ODE solution using the Fourth-Order Adams-Moulton Scheme
        	 and name of the scheme
        """
        I = np.size(self.t)
        y = np.zeros((self.size, I))
        one = np.ones((self.size, I))
        initial_step, name = Scheme.runge_kutta(self)
        for i in range(0, I):
            if i == 0:
                y[:, i] = self.y0[:]
            elif i < 3:
                y[:, i] = initial_step[:, i]
            else:
                k1 = self.f(self.t[i-1], y[:, i-1])
                k2 = self.f(self.t[i-1] + self.h/2, y[:, i-1] + k1*one[:,i]*self.h/2)
                k3 = self.f(self.t[i-1] + self.h/2, y[:, i-1] + k2*one[:,i]*self.h/2)
                k4 = self.f(self.t[i-1] + self.h, y[:, i-1] + k2*one[:,i]*self.h)
                y[:, i] = y[:, i-1] + self.h/6 * (k1 + 2*k2 + 2*k3 + k4)
                y[:, i] = y[:, i - 1] + self.h / 24 * (9 * self.f(self.t[i], y[:, i]) +
                                                       19 * self.f(self.t[i - 1], y[:, i - 1]) -
                                                       5 * self.f(self.t[i - 2], y[:, i - 2]) +
                                                       self.f(self.t[i - 3], y[:, i - 3]))
        return y, 'adams moulton 4'

    def backward_euler(self):
        """
        :return: numerical approximation of ODE solution using the Backward Euler Scheme
        	 and name of the scheme
        """
        I = np.size(self.t)
        y = np.zeros((self.size,I))
        for i in range(0, I):
            if i == 0:
                y[:, i] = self.y0[:]
            else:
                y[:, i] = y[:, i-1] + self.h * self.f(self.t[i-1], y[:, i-1])
                y[:, i] = y[:, i-1] + self.h * self.f(self.t[i], y[:, i])
        return y, 'backward euler'

    def euler(self):
        """
        :return: numerical approximation of ODE solution using the Euler Scheme
        	 and name of the scheme
        """
        I = np.size(self.t)
        y = np.zeros((self.size,I))
        for i in range(0, I):
            if i == 0:
                y[:, i] = self.y0[:]
            else:
                y[:, i] = y[:, i-1] + self.h * self.f(self.t[i-1], y[:, i-1])
        return y, 'euler'

    def exact_solution(self, ODE_analytic_solution):
        """
        :param ODE_analytic_solution: mathematical function representing the analytic solution of the ODE.
        :return: 		      exact solution of the ODE
        """
        I = np.size(self.t)
        exact = np.zeros((self.size, I))
        for i in range(0, I):
            exact[:, i] = ODE_analytic_solution(self.t[i], self.y0)
        return exact

    def heun(self):
        """
        :return: numerical approximation of ODE solution using the Improved Euler (Heun) Scheme
        	 and name of the scheme
        """
        I = np.size(self.t)
        y = np.zeros((self.size, I))
        for i in range(0, I):
            if i == 0:
                y[:, i] = self.y0[:]
            else:
                y[:, i] = y[:, i-1] + self.h * self.f(self.t[i-1], y[:, i-1])
                y[:, i] = y[:, i-1] + self.h/2 * (self.f(self.t[i-1], y[:, i-1]) +
                                                  self.f(self.t[i], y[:, i]))
        return y, 'heun (improved euler)'

    def runge_kutta(self):
        """
        :return: numerical approximation of ODE solution using the Runge-Kutta Scheme
        	 and name of the scheme
        """
        I = np.size(self.t)
        y = np.zeros((self.size, I))
        one = np.ones((self.size,I))
        for i in range(0, I):
            if i == 0:
                y[:,i] = self.y0[:]
            else:
                k1 = self.f(self.t[i-1], y[:, i-1])
                k2 = self.f(self.t[i-1] + self.h/2, y[:, i-1] + k1*one[:,i]*self.h/2)
                k3 = self.f(self.t[i-1] + self.h/2, y[:, i-1] + k2*one[:,i]*self.h/2)
                k4 = self.f(self.t[i-1] + self.h, y[:, i-1] + k3*one[:,i]*self.h)
                y[:, i] = y[:, i-1] + self.h/6 * (k1 + 2*k2 + 2*k3 + k4)
        return y, 'runge-kutta'

class SDE(Scheme):
    def __init__(self,
                 differential_equation,
                 stochastic,
                 initial_condition,
                 start_time,
                 stop_time,
                 time_step,
                 mean = 0,
                 standard_deviation = 1):
        self.g = stochastic
        self.mean = mean
        self.std = standard_deviation
        super().__init__(
                 differential_equation,
                 initial_condition,
                 start_time,
                 stop_time,
                 time_step)

    def euler_mayurama(self):
        I = np.size(self.t)
        y = np.zeros((self.size, I))
        noise = np.random.normal(self.mean, self.std, size=self.t.size)
        for i in range(0, I):
            if i == 0:
                y[:, i] = self.y0[:]
            else:
                y[:, i] = y[:, i - 1] + self.f(self.t[i - 1], y[:, i - 1])*self.h + \
                                        self.g(self.t[i - 1], y[:, i - 1])*np.sqrt(self.h)*noise[i-1]
        return y, 'euler-mayurama'

class Grid:
    """
    methods to define intervals at which to collect information from a model
    """

    def __init__(self, start_time, stop_time, time_step):
        """
        :param start_time: 	      start time of the measurement expressed as an integer
        :param stop_time: 	      stop time of the measurement expressed as an integer
        :param time_step: 	      time step/resolution of the scheme's grid expressed as an integer
        			      appears as h or dt in analytic work
        """
        self.x_min = start_time
        self.x_max = stop_time
        self.h = time_step

    def create_grid(self):
        """
        :return: an array to be passed as input data in __init__
        """
        x = np.zeros(int((self.x_max - self.x_min) / self.h))
        x[0] = self.x_min
        for i in range(1, x.size):
            x[i] = x[i-1] + self.h
        x = np.append(x, self.x_max)
        return x

class Error:
    """
    a collection of different methods to calculate error
    """

    def __init__(self, exact_solution, approximate_solution):
        """
        :param exact_solution: 	     array of the exact solution at a point. Can use Scheme.exact_solution() as input
        :param approximate_solution: array of the approximate solution. Use Scheme as input
        :param system_size: 	     number of equations in system expressed as an integer
        """
        self.exact = exact_solution
        self.approx = approximate_solution
        self.size = np.size(exact_solution[:, 0])

    def relative_error(self):
        """
        :return: relative error as a decimal, multiply by 100 for percent error
        """
        I = np.size(self.exact[0, :])
        error = np.zeros((self.size, I))
        for i in range(0, I):
            error[:, i] = np.abs((self.exact[:, i] - self.approx[:, i])/self.exact[:, i])
        return error
    
class Systems:
    def __init__(self, data_array):
        self.X = data_array

    def lorenz(self, sigma=10, r=28, b=8/3):
        x_dot = sigma*(self.X[1] - self.X[0])
        y_dot = self.X[0]*(r - self.X[2]) - self.X[1]
        z_dot = self.X[0]*self.X[1] - b*self.X[2]
        return np.array([x_dot, y_dot, z_dot])

    def rossler(self, a=0.2, b=0.2, c=5.7):
        x_dot = -self.X[1] - self.X[2]
        y_dot = self.X[0] + a*self.X[1]
        z_dot = b + self.X[2]*(self.X[0] - c)
        return np.array([x_dot, y_dot, z_dot])

    def chua(self, A=9, B=100/7):
        #function = lambda x: m1*x + 0.5*(m0 - m1)*(np.abs(x + 1) - np.abs(x - 1))
        def func(x):
            a=-8/7
            b=-5/7
            if x >= 1:
                return b*x+a-b
            elif x < 1 and x > -1:
                return a*x
            else:
                return b*x-a+b
        x_dot = A*(self.X[1] - self.X[0] - func(self.X[0]))
        y_dot = self.X[0] - self.X[1] + self.X[2]
        z_dot = -B*self.X[1]
        return np.array([x_dot, y_dot, z_dot])

    def signum(self, a=1):
        x_dot = self.X[1] - self.X[0]
        y_dot = -self.X[2]*np.sign(self.X[0])
        z_dot = self.X[0]*np.sign(self.X[1]) - a
        return np.array([x_dot, y_dot, z_dot])

    def rabinovich(self, a=.98, b=.1):
        x_dot = self.X[1]*(self.X[2] - 1 + self.X[0]**2) + b*self.X[0]
        y_dot = self.X[0]*(3*self.X[2] + 1 - self.X[0]**2) + b*self.X[1]
        z_dot = -2*self.X[2]*(a + self.X[0]*self.X[1])
        return np.array([x_dot, y_dot, z_dot])

    def gyorgyi_fields(self, kf=3.5e-4):
        # ------------------
        # constants
        # X0 = np.array([2, 5, 3.385])
        # ------------------
        kGF1 = 4.0e6
        kGF2 = 2.0
        kGF3 = 3000
        kGF4 = 55.2
        kGF5 = 7000
        kGF6 = 0.09
        kGF7 = 0.23
        M = 0.25
        A = 0.1
        H = 0.26
        C = 8.33e-4
        # kf = 3.5e-4#2.16e-3
        alpha = 666.7
        beta = 0.3478
        X0 = 1.93e-6 * M
        Z0 = 8.33e-6 * M
        V0 = 1.39e-3 * M
        Y0 = 7.72e-6 * M
        T0 = 2308.6
        x, z, v = self.X
        y_til = ((alpha * kGF6 * Z0 * V0 * z * v) / (kGF1 * H * X0 * x + kGF2 * A * H ** 2 + kf)) / Y0
        x_dot = T0 * (-kGF1 * H * Y0 * x * y_til
                      + (kGF2 * A * H ** 2 * Y0 / X0) * y_til
                      - 2 * kGF3 * X0 * x ** 2 + 0.5 * kGF4 * A ** (0.5) * H ** (1.5) * X0 ** (-0.5) * (
                                  C - Z0 * z) * x ** (
                          0.5)
                      - 0.5 * kGF5 * Z0 * x * z - kf * x)  # added 0.5 to kGF5
        z_dot = T0 * (kGF4 * A ** (0.5) * H ** (1.5) * X0 ** (0.5) * (C / Z0 - z) * x ** (0.5)
                      - kGF5 * X0 * x * z
                      - alpha * kGF6 * V0 * z * v
                      - beta * kGF7 * M * z - kf * z)
        v_dot = T0 * ((2 * kGF1 * H * X0 * Y0 / V0) * x * y_til
                      + (kGF2 * A * H ** 2 * Y0 / V0) * y_til
                      + (kGF3 * X0 ** 2 / V0) * x ** 2
                      - alpha * kGF6 * Z0 * z * v - kf * v)
        system = np.array([x_dot, z_dot, v_dot])
        return system

if __name__=="__main__":
    start_time = 0
    stop_time = 2
    time_step = 0.00001

    x = Grid(start_time, stop_time, time_step).create_grid()
    init_cond = X0 = np.array([2, 5, 3.385])
    system = lambda t, X: Systems(X).gyorgyi_fields()
    solver = Scheme(system, init_cond, start_time, stop_time, time_step)
    rawdata, name = solver.adams_moulton4()

    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")
    ax.plot3D(rawdata[0, :], rawdata[1, :], rawdata[2, :])
    plt.show()





