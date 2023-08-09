import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from invertedPendulum import InvertedPendulum


# Define the InvertedPendulum class here
# ...

# Model Parameters
params = {
    'length': 1,      # [m]        length of pendulum
    'mass': 1,        # [kg]       mass of pendulum
    'gravity': 9.81,  # [m/s^2]    acceleration of gravity
    'friction': 0.01,  # [s*Nm/rad] friction coefficient
    'u_max': 100,
    'u_min': -100,
    'I': 1/3,  # Assumed moment of inertia
    'Kp': 6,
    'Kd': 5,
    'clf': {
        'rate': 3,
    },
    'weight': {
        'slack': 100000,
    }
}

x0 = np.array([2, 0.05])  # Initial state

ip_sys = InvertedPendulum(params)

controller = ip_sys.ctrl_clf_qp

# Simulation settings
dt = 0.02
sim_t = 10
total_k = int(np.ceil(sim_t / dt))

# Initialize traces
xs = np.zeros((total_k, ip_sys.xdim))
ts = np.zeros(total_k)
us = np.zeros((total_k - 1, 1))
Vs = np.zeros(total_k - 1)

# Initial state
x = x0
t = 0

# Store initial state
xs[0] = x0
ts[0] = t

# Simulation loop
for k in range(total_k - 1):
    # Determine control input.
    u,slack,V,feas,comp = controller(x)
    print(u)
    us[k] = u
    # Run one time step propagation.
    sol = solve_ivp(lambda t, s: ip_sys.dynamics(t, s, u), [t, t + dt], x, t_eval=[t + dt])
    x = sol.y[:, -1]
    xs[k + 1] = x
    ts[k + 1] = sol.t[-1]
    t += dt

# Plotting results
plt.figure()
plt.suptitle('Inverted Pendulum: CLF-QP States')

plt.subplot(2, 1, 1)
plt.plot(ts, 180 * xs[:, 0] / np.pi)
plt.xlabel("t (sec)")
plt.ylabel("theta (deg)")

plt.subplot(2, 1, 2)
plt.plot(ts, 180 * xs[:, 1] / np.pi)
plt.xlabel("t (sec)")
plt.ylabel("dtheta (deg/s)")

plt.figure()
plt.plot(ts[:-1], us)
plt.plot(ts[:-1], params['u_max'] * np.ones(total_k - 1), 'k--')
plt.plot(ts[:-1], params['u_min'] * np.ones(total_k - 1), 'k--')
plt.xlabel("t (sec)")
plt.ylabel("u (N.m)")

plt.show()
