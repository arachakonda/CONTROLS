from controlAffine import CtrlAffineSys
from sympy import Matrix
import sympy as sym


class ACC(CtrlAffineSys):
    def __init__(self, params):
        super().__init__(params)
        self.params = params

    def define_clf(self, symbolic_state):
        v = symbolic_state[1]
        vd = self.params['vd']
        clf = Matrix([(v - vd)**2])
        return clf

    
    def define_system(self, params):
        # Define the system dynamics for the inverted pendulum
        # Inputs: params - dictionary containing system parameters
        # Outputs: x - symbolic state vector
        #          f - drift term, expressed symbolically with respect to x
        #          g - control vector fields, expressed symbolically with respect to x
        p, v, z = sym.symbols('p v z')
        x = Matrix([p, v, z])
        f0 = params['f0']
        f1 = params['f1']
        f2 = params['f2']
        v0 = params['v0']
        m = params['m']
        Fr = f0 + f1 * v + f2 * v**2

        f = Matrix([v, -Fr/m, v0-v])
        g = Matrix([0, 1/m, 0])


        return x, f, g