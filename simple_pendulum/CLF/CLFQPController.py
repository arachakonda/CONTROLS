import numpy as np
from sympy import symbols, lambdify
from sympy import Matrix
from sympy import sin, cos
import time
import cvxpy as cp
import scipy
class ControlAffineSystem:
    def __init__(self,params):
        self.xdim = None
        self.udim = None
        self.params = params
        self.f = None
        self.g = None
        self.clf = None
        self.lf_clf = None
        self.lg_clf = None
        self.clf_rate = self.params['clf_rate']
        self.weight_input = self.params['weight_input']
        self.u_max = self.params['u_max']
        self.u_min = self.params['u_min']

        x,f,g = self.defineSystem(params)
        self.f = f
        self.g = g
        clf= self.defineClf(params, x)
        self.clf = clf
        self.initSys(x,f,g,clf)
    
    def defineClf(self, params, sym_x):
        clf = None
        return clf
    def defineSystem(self, params):
        x = None
        f = None
        g = None
        return x,f,g
    def dynamics(self,t,x,u):
        dx = self.f(x) + self.g(x) * u
        return dx
    
    def initSys(self, sym_x, sym_f, sym_g, sym_clf):
        if sym_x is None or sym_f is None or sym_g is None:
            raise ValueError('x, f, g is empty. Create a class function defineSystem and define your dynamics with symbolic expression.')
        
        # if the sym_f, sym_g are not in the form of sympy.Matrix, convert them
        if not isinstance(sym_f, Matrix):
            f_ = Matrix(sym_f)
        else:
            f_ = sym_f
        if not isinstance(sym_g, Matrix):
            g_ = Matrix(sym_g)
        else:
            g_ = sym_g
            x  = sym_x

        #setting state and input dimension
        self.xdim = len(x)
        self.udim = len(g_.T)

        #setting f and g as a function
        self.f = lambdify((x), f_, modules='numpy')
        self.g = lambdify((x), g_, modules='numpy')


        #obtaining the lie derivatives of the CLF
        #if sym_clf is not empty, then calculate the lie derivatives and simplify them
        if sym_clf is not None:
            #evalute jacobian wrt x and simplify
            d_clf = sym_clf.jacobian(x)
            lf_clf_ = d_clf*f_
            lg_clf = d_clf*g_
            self.clf = lambdify((x), sym_clf, modules='numpy')
            self.lf_clf = lambdify((x), lf_clf_, modules='numpy')
            self.lg_clf = lambdify((x), lg_clf, modules='numpy')

    

    #Implementation of vanilla CLF-QP
    #Inputs:   x: state
    #          u_ref: reference control input
    #          with_slack: flag for relaxing (1: relax, 0: hard CLF constraint)
    #          verbose: flag for logging (1: print log, 0: run silently)
    #Outputs:  u: control input as a solution of the CLF-QP
    #          slack: slack variable for relaxation. (empty list when with_slack=0)
    #          V: Value of the CLF at the current state.
    #          feas: 1 if QP is feasible, 0 if infeasible. (Note: even
    #          when qp is infeasible, u is determined from quadprog.)
    #          comp_time: computation time to run the solver.
    def clf_qp(self, x, u_ref, with_slack=0):
        # if the clf is empty raise error that it cannot be used
        if self.clf is None:
            raise ValueError('clf is empty. Create a class function defineSystem and define your dynamics with symbolic expression.')
        #if the u_ref is empty, set it to zero
        if u_ref is None:
            u_ref = np.zeros(self.u_dim)
        #if the with_slack is empty, set it to 1
        if with_slack is None:
            with_slack = 1
        
        #if the length of u_ref is not equal to u_dim, raise error
        if len(u_ref) != self.u_dim:
            raise ValueError('u_ref is not the same dimension as the input dimension of the system.')

        tstart = time.time()
        V = self.clf(x)
        lf_V = self.lf_clf(x)
        lg_V = self.lg_clf(x)

        #constraints : A[u;slack] <= b
        if with_slack:
            #CLF constraint
            A = np.hstack((lg_V, -np.ones((lg_V.shape[0], 1))))
            b = -lf_V - self.clf_rate*V
            # add input contstraint if u_max and u_min are defined
            if self.u_max is not None:
                A = np.vstack((A, np.hstack((np.eye(self.udim), np.zeros((self.udim, 1))))))
                if self.u_max.shape[0] == 1:
                    b = np.vstack((b, np.tile(self.u_max, (self.udim, 1))))
                elif self.u_max.shape[0] == self.udim:
                    b = np.vstack((b, self.u_max))
                else:
                    raise ValueError("u_max should be either a scalar value or an (udim, 1) array.")
            if self.u_min is not None:
                A = np.vstack((A, -np.eye(self.udim), np.zeros((self.udim, 1))))
                if self.u_min.shape[0] == 1:
                    b = np.vstack((b, -self.u_min * np.ones((self.udim, 1))))
                elif self.u_min.shape[0] == self.udim:
                    b = np.vstack((b, -self.u_min))
                else:
                    raise ValueError("u_min should be either a scalar value or an (udim, 1) array.")
        else:
            A = lg_V
            b = -lf_V - self.clf_rate*V
            # add input contstraint if u_max and u_min are defined
            if self.u_max is not None:
                A = np.vstack((A, np.eye(self.udim)))
                if self.u_max.shape[0] == 1:
                    b = np.vstack((b, np.tile(self.u_max, (self.udim, 1))))
                elif self.u_max.shape[0] == self.udim:
                    b = np.vstack((b, self.u_max))
                else:
                    raise ValueError("u_max should be either a scalar value or an (udim, 1) array.")
            if self.u_min is not None:
                A = np.vstack((A, -np.eye(self.udim)))
                if self.u_min.shape[0] == 1:
                    b = np.vstack((b, -self.u_min * np.ones((self.udim, 1))))
                elif self.u_min.shape[0] == self.udim:
                    b = np.vstack((b, -self.u_min))
                else:
                    raise ValueError("u_min should be either a scalar value or an (udim, 1) array.")
            
        #cost function: 1/2*[u;slack]'*H*[u;slack] + f'*[u;slack]
        # H = [weight_input, zeros(obj.udim, 1);
        #     zeros(1, obj.udim), obj.params.weight_slack];
        # f_ = [-weight_input * u_ref; 0];
        H = np.zeros((self.u_dim+1, self.u_dim+1))
        H[:self.u_dim, :self.u_dim] = self.weight_input
        H[self.u_dim, self.u_dim] = self.params.weight_slack
        f_ = np.zeros((self.u_dim+1, 1))
        f_[:self.u_dim] = -self.weight_input * u_ref
        # solve QP
        u, slack, feas = self.solve_qp(H, f_, A, b)
        # if the QP is infeasible, set u to the solution of the unconstrained QP
        if feas == 0:
            u, slack, feas = self.solve_qp(H, f_, None, None)
        # if the QP is still infeasible, set u to the reference input
        if feas == 0:
            u = u_ref
        # if the QP is feasible, but the slack variable is negative, set u to the reference input
        elif slack < 0:
            u = u_ref
        # if the QP is feasible, but the slack variable is positive, set u to the solution of the unconstrained QP
        elif slack > 0:
            u, slack, feas = self.solve_qp(H, f_, None, None)
            # if the QP is infeasible, set u to the reference input
            if feas == 0:
                u = u_ref
        # if the QP is feasible, but the slack variable is zero, do nothing
        else:
            pass
        # compute the computation time
        comp_time = time.time() - tstart
        return u, feas, comp_time
    
    # % solve_qp: solve a quadratic program with linear inequality constraints using cvxpy
    def solve_qp(self, H, f, A, b):
        # define the variable
        u = cp.Variable((self.u_dim, 1))
        slack = cp.Variable((1, 1))
        # define the cost function
        cost = 0.5 * cp.quad_form(u, H) + f.T @ u
        # define the constraints
        constraints = []
        if A is not None:
            constraints += [A @ cp.vstack((u, slack)) <= b]
        # define the problem
        prob = cp.Problem(cp.Minimize(cost), constraints)
        # solve the problem
        prob.solve()
        # return the solution
        return u.value, slack.value, prob.status


class pendulum(ControlAffineSystem):
    def __init__(self, params):
        super().__init__(params)
        self.params = params

    def defineSystem(self, params):
        theta, dtheta = symbols('theta dtheta')
        x = Matrix([theta, dtheta])
        l = self.params['l']    # [m]        length of pendulum
        m = self.params['m']    # [kg]       mass of pendulum
        g = self.params['g']    # [m/s^2]    acceleration of gravity
        b = self.params['b']    # [s*Nm/rad] friction coefficient
        # f = [x(2); (-b*x(2) + m*g*l*sin(x(1))/2 ) / (m*l^2/3)];
        # g = [0; -1 / (m*l^2/3)];
        f = Matrix([x[1], (-b*x[1] + m*g*l*sin(x[0])/2) / (m*l**2/3)])
        g = Matrix([0, -1 / (m*l**2/3)])
        return x,f,g

    def defineClf(self,params,sym_x):
        x = sym_x
        clf = x.T @ x
        return clf
    def desired_trajectory(self,t):
        theta_desired = np.pi / 2 * np.sin(t)
        theta_dot_desired = np.pi / 2 * np.cos(t)
        theta_double_dot_desired = -(np.pi / 2) * np.sin(t)
        return theta_desired, theta_dot_desired, theta_double_dot_desired
    
dt = 0.02
sim_t = 5

params = {'l': 0.5, 'm': 0.5, 'g': 9.81, 'b': 0.1, 'u_max': 7, 'u_min': -7, 'Kp' : 6, 'Kd' : 5, 'clf_rate': 3, 'weight_slack': 1e-4, 'weight_input': 1e+4}
pendulum = pendulum(params)
x0 = np.array([0.1, 0.1])
ode = pendulum.dynamics
controller = pendulum.clf_qp
solver = scipy.integrate.solve_ivp #by default RK45

total_k = int(sim_t/dt)
x = x0
t = 0

#initialize the data

xs = np.zeros((total_k, pendulum.xdim))
ts = np.zeros((total_k, 1))
us = np.zeros((total_k-1, 1))
vs = np.zeros((total_k-1, 1))
xs[0, :] = x0
ts[0] = t

for k in range(total_k-1):
    #determine the control input
    # dV_hat: analytic Vdot based on model
    print(t)
    theta_desired, theta_dot_desired, theta_double_dot_desired = pendulum.desired_trajectory(t)
    uref = -pendulum.params['Kp'] * (x[0] - theta_desired) - pendulum.params['Kd'] * (x[1] - theta_dot_desired)
    u, slack, V = controller(x, )
    us[k,:] = u
    vs[k] = V

    # Run one time step propagation.
    t_temp, x_temp, _, _, _, _, _, _, _, _, _  = solver(ode, [t, t+dt], x, method='RK45', rtol=1e-6, atol=1e-6)

    x = x_temp[-1,:].T
    xs[k+1,:] = x.T
    ts[k+1] = t_temp[-1]
    t = t+dt

# Plot the results
plt.figure()
plt.plot(ts, 180*s[:,0]/np.pi, label='theta')
plt.xlabel('time [s]')
plt.ylabel('theta [deg]')

plt.figure()
plt.plot(ts, 180*s[:,1]/np.pi, label='dtheta')
plt.xlabel('time [s]')
plt.ylabel('dtheta [deg/s]')

plt.figure()
plt.plot(ts[:-1], us, label='u')
plt.xlabel('time [s]')
plt.ylabel('u [Nm]')
plt.show()





