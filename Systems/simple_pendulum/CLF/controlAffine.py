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
        #if length of x is 2 then return dx accordingly
        if len(x) == 2:
            dx = self.f(x[0],x[1]) + self.g(x[0],x[1]) * u
        elif len(x) == 3:
            dx = self.f(x[0],x[1],x[2]) + self.g(x[0],x[1],x[2]) * u
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
            self.clf = lambdify(x, symbolic_clf, 'numpy')
            self.lf_clf = lambdify(x, lf_clf_, 'numpy')
            # TODO: add sanity check of relative degree.
            self.lg_clf = lambdify(x, lg_clf_, 'numpy')
    
    def ctrl_clf_qp(self, x, u_ref=None, with_slack=True, verbose=False):
        if not hasattr(self, "clf"):
            raise ValueError("CLF is not defined. Ensure 'clf' is defined in the input object.")

        udim = self.udim

        if u_ref is None:
            u_ref = np.zeros((udim, 1))

        if u_ref.shape[0] != udim:
            raise ValueError("Wrong size of u_ref, it should be (udim, 1) array.")

        if len(x) == 2:
            V = self.clf(x[0], x[1])
            LfV = self.lf_clf(x[0], x[1])
            LgV = self.lg_clf(x[0], x[1])
        elif len(x) == 3:
            V = self.clf(x[0], x[1], x[2])
            LfV = self.lf_clf(x[0], x[1], x[2])
            LgV = self.lg_clf(x[0], x[1], x[2])

        # Declare the variable 'u' here, before using it in the constraints
        u = cp.Variable((udim, 1))
        constraints = []

        if with_slack:
            slack = cp.Variable()
            constraints.append(LgV @ u + slack <= -LfV - self.params["clf"]["rate"] * V)

            if "u_max" in self.params:
                u_max = np.array(self.params["u_max"])
                if u_max.size == 1:
                    u_max = u_max * np.ones((udim, 1))
                constraints.append(u <= u_max)

            if "u_min" in self.params:
                u_min = np.array(self.params["u_min"])
                if u_min.size == 1:
                    u_min = u_min * np.ones((udim, 1))
                constraints.append(u >= u_min)
        else:
            constraints.append(LgV @ u <= -LfV - self.params["clf"]["rate"] * V)

            if "u_max" in self.params:
                u_max = np.array(self.params["u_max"])
                if u_max.size == 1:
                    u_max = u_max * np.ones((udim, 1))
                constraints.append(u <= u_max)

            if "u_min" in self.params:
                u_min = np.array(self.params["u_min"])
                if u_min.size == 1:
                    u_min = u_min * np.ones((udim, 1))
                constraints.append(u >= u_min)

        # Cost formulation
        if "weight" in self.params and "input" in self.params["weight"]:
            weight_input = np.array(self.params["weight"]["input"])
            if weight_input.size == 1:
                weight_input = weight_input * np.eye(udim)
        else:
            weight_input = np.eye(udim)

        cost = cp.quad_form(u - u_ref, weight_input)

        if with_slack:
            print(self.params["weight"]["slack"])
            cost += self.params["weight"]["slack"] * cp.norm(slack)

        prob = cp.Problem(cp.Minimize(cost), constraints)

        if verbose:
            prob.solve(verbose=True)
        else:
            prob.solve()

        feas = 1 if prob.status == cp.OPTIMAL else 0

        if with_slack:
            return u.value, slack.value, V, feas
        else:
            return u.value, [], V, feas
