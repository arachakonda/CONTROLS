import control as ct
from sympy import Matrix
import numpy as np
import sympy as sym
from controlAffine import CtrlAffineSys

ct.use_numpy_matrix(True, False)

class InvertedPendulum(CtrlAffineSys):
    def __init__(self, params):
        super().__init__(params)
        self.params = params

    def define_clf(self, symbolic_state):
        # Define the Control Lyapunov Function (CLF) for the inverted pendulum
        # Inputs: params - dictionary containing system parameters
        #         symbolic_state - symbolic state vector created in define_system
        # Outputs: clf - CLF expressed symbolically with respect to symbolic_state
        x = symbolic_state
        I = self.params['mass'] * self.params['length'] ** 2 / 3
        c_bar = self.params['mass'] * self.params['gravity'] * self.params['length'] / (2 * I)
        b_bar = self.params['friction'] / I
        A = np.array([[0, 1], 
                        [c_bar - self.params['Kp'] / I, -b_bar - self.params['Kd'] / I]])  # Linearized Dynamics.
        Q = self.params['clf']['rate'] * np.eye(A.shape[0])
        P = ct.lyap(A, Q)  # Cost Matrix for quadratic CLF. (V = e' * P * e)
        clf = x.T @ P @ x
        return clf
    
    def define_system(self, params):
        # Define the system dynamics for the inverted pendulum
        # Inputs: params - dictionary containing system parameters
        # Outputs: x - symbolic state vector
        #          f - drift term, expressed symbolically with respect to x
        #          g - control vector fields, expressed symbolically with respect to x
        theta, dtheta = sym.symbols('theta dtheta')
        x = Matrix([theta, dtheta])
        l = params['length']  # [m]        length of the pendulum
        m = params['mass']    # [kg]       mass of the pendulum
        g = params['gravity']  # [m/s^2]    acceleration due to gravity
        b = params['friction']  # [s*Nm/rad] friction coefficient

        f = Matrix([x[1], (-b * x[1] + m * g * l * sym.sin(x[0]) / 2) / (m * l ** 2 / 3)])
        g = Matrix([0, -1 / (m * l ** 2 / 3)])

        return x, f, g