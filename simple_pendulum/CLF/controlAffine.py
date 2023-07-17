from sympy import Matrix, simplify, lambdify
import sympy as sym
import numpy as np
import time
import cvxpy as cp

class CtrlAffineSys:
    def __init__(self, params=None):
        self.xdim = None  # State dimension
        self.udim = None  # Control input dimension
        self.params = params  # Model parameters as a dictionary object
        
        self.f = None  # Drift term in the dynamics as a function
        self.g = None  # System vector fields in the dynamics as a function
        
        self.cbf = None  # Control Barrier Function as a function
        self.lf_cbf = None  # Lie derivative (wrt f) of the CBF as a function
        self.lg_cbf = None  # Lie derivative (wrt g) of the CBF as a function
        self.clf = None  # Control Lyapunov Function as a function
        self.lf_clf = None  # Lie derivative (wrt f) of the CLF as a function
        self.lg_clf = None  # Lie derivative (wrt g) of the CLF as a function
        
        if params is None:
            self.params = {}
            print("Warning: params argument is missing.")
        
        # TODO: Add checking input constraint.
        # TODO: Add existence of essential fields (e.g., params['weight']['slack']).
        x, f, g = self.define_system(self.params)
        clf = self.define_clf(x)
        cbf = self.define_cbf(x)
        self.init_sys(x, f, g, cbf, clf)

    def define_system(self, params):
        # Outputs: x: symbolic state vector
        #          f: drift term, expressed symbolically with respect to x.
        #          g: control vector fields, expressed symbolically with respect to x.
        x = None
        f = None
        g = None
        return x, f, g

    def define_clf(self,symbolic_state):
        # symbolic_state: same symbolic state created in define_system
        # clf: CLF expressed symbolically with respect to symbolic_state.
        clf = None
        return clf

    def define_cbf(self,symbolic_state):
        # symbolic_state: same symbolic state created in define_system
        # cbf: CBF expressed symbolically with respect to symbolic_state.
        cbf = None
        return cbf

    def dynamics(self, t, x, u):
        # Inputs: t: time, x: state, u: control input
        # Output: dx: xdot
        dx = self.f(x[0],x[1]) + self.g(x[0],x[1]) * u
        return dx
    
    def init_sys(self, symbolic_x, symbolic_f, symbolic_g, symbolic_cbf, symbolic_clf):
        if symbolic_x is None or symbolic_f is None or symbolic_g is None:
            raise ValueError('x, f, g is empty. Create a class function defineSystem and define your dynamics with symbolic expression.')

        if not isinstance(symbolic_f, Matrix):
            f_ = Matrix(symbolic_f)
        else:
            f_ = symbolic_f

        if not isinstance(symbolic_g, Matrix):
            g_ = Matrix(symbolic_g)
        else:
            g_ = symbolic_g

        if not isinstance(symbolic_x, Matrix):
            x = Matrix(symbolic_x)
        else:
            x = symbolic_x
        # Setting state and input dimension.
        self.xdim = x.shape[0]
        self.udim = g_.shape[1]
        # Setting f and g (dynamics)
        self.f = lambdify(x, f_)
        self.g = lambdify(x, g_)

        # Obtaining Lie derivatives of CBF.
        if symbolic_cbf is not None:
            dcbf = symbolic_cbf.jacobian(symbolic_x)
            lf_cbf_ = dcbf * f_
            lg_cbf_ = dcbf * g_
            self.cbf = lambdify(x, symbolic_cbf)
            self.lf_cbf = lambdify(x, lf_cbf_)
            # TODO: add sanity check of relative degree.
            self.lg_cbf = lambdify(x, lg_cbf_)

        # Obtaining Lie derivatives of CLF.
        if symbolic_clf is not None:
            dclf = sym.simplify(symbolic_clf.jacobian(symbolic_x))
            lf_clf_ = dclf * f_
            lg_clf_ = dclf * g_
            print("symbolic_clf: ",symbolic_clf)
            print("x : ",x)
            self.clf = lambdify([x[0],x[1]], symbolic_clf, 'numpy')
            self.lf_clf = lambdify([x[0],x[1]], lf_clf_, 'numpy')
            # TODO: add sanity check of relative degree.
            self.lg_clf = lambdify([x[0],x[1]], lg_clf_, 'numpy')
    
    def ctrl_clf_qp(self, x, u_ref=None, with_slack=True, verbose=False):
        if self.clf is None:
            raise ValueError('CLF is not defined so ctrlClfQp cannot be used. Create a class function [defineClf] and set up clf with symbolic expression.')

        if u_ref is None:
            u_ref = np.zeros((self.udim, 1))

        if u_ref.shape != (self.udim, 1):
            raise ValueError("Wrong size of u_ref, it should be (udim, 1) array.")

        tstart = time.time()
        V = self.clf(x[0], x[1])
        LfV = self.lf_clf(x[0], x[1])
        LgV = self.lg_clf(x[0], x[1])

        if with_slack:
            # CLF constraint.
            slack_column = np.full((LgV.shape[0], 1), -1)
            A = np.hstack((LgV, slack_column))
            b = -LfV - self.params['clf']['rate'] * V

            # Input constraints.
            if hasattr(self.params, 'u_max'):
                A = np.vstack((A, np.hstack((np.eye(self.udim), np.zeros((self.udim, 1))))))
                if self.params['u_max'].size == 1:
                    b = np.vstack((b, self.params['u_max'] * np.ones((self.udim, 1))))
                elif self.params['u_max'].size == self.udim:
                    b = np.vstack((b, self.params['u_max']))
                else:
                    raise ValueError("params.u_max should be either a scalar value or an (udim, 1) array.")

            if hasattr(self.params, 'u_min'):
                A = np.vstack((A, np.hstack((-np.eye(self.udim), np.zeros((self.udim, 1))))))
                if self.params['u_min'].size == 1:
                    b = np.vstack((b, -self.params['u_min'] * np.ones((self.udim, 1))))
                elif self.params['u_min'].size == self.udim:
                    b = np.vstack((b, -self.params['u_min']))
                else:
                    raise ValueError("params.u_min should be either a scalar value or an (udim, 1) array.")
        else:
            # CLF constraint.
            A = LgV
            b = -LfV - self.params['clf']['rate'] * V

            # Input constraints.
            if hasattr(self.params, 'u_max'):
                A = np.vstack((A, np.eye(self.udim)))
                if self.params['u_max'].size == 1:
                    b = np.vstack((b, self.params['u_max'] * np.ones((self.udim, 1))))
                elif self.params['u_max'].size == self.udim:
                    b = np.vstack((b, self.params['u_max']))
                else:
                    raise ValueError("params.u_max should be either a scalar value or an (udim, 1) array.")

            if hasattr(self.params, 'u_min'):
                A = np.vstack((A, -np.eye(self.udim)))
                if self.params['u_min'].size == 1:
                    b = np.vstack((b, -self.params['u_min'] * np.ones((self.udim, 1))))
                elif self.params['u_min'].size == self.udim:
                    b = np.vstack((b, -self.params['u_min']))
                else:
                    raise ValueError("params.u_min should be either a scalar value or an (udim, 1) array.")

        # Cost
        if hasattr(self.params['weight'], 'input'):
            if self.params['weight']['input'].size == 1:
                weight_input = self.params['weight']['input'] * np.eye(self.udim)
            elif self.params['weight']['input'].shape == (self.udim, self.udim):
                weight_input = self.params['weight']['input']
            else:
                raise ValueError("params.weight.input should be either a scalar value or an (udim, udim) array.")
        else:
            weight_input = np.eye(self.udim)

        H = np.block([[weight_input, np.zeros((self.udim, 1))],
                    [np.zeros((1, self.udim)), self.params['weight']['slack']]])
        f_ = np.vstack([-np.dot(weight_input, u_ref), np.zeros((1, 1))])

        # Define optimization variables.
        z = cp.Variable(self.udim + 1)

        # Define the objective and constraints.
        objective = cp.Minimize(0.5 * cp.quad_form(z, H) + f_.T @ z)
        constraints = [A @ z <= b]

        # Create the problem and solve it.
        problem = cp.Problem(objective, constraints)
        result = problem.solve()

        # Extract the optimal control input.
        u_slack = z.value
        u = u_slack[:self.udim]

        # Check if the problem was feasible.
        feas = problem.status == cp.OPTIMAL

        slack = u_slack[-1] if with_slack else []
        comp_time = time.time() - tstart

        return u, slack, V, feas, comp_time
