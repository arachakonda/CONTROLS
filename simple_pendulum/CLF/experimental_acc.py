import numpy as np
from ACC import ACC
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
    


params = {
    'v0': 14,
    'vd': 24,
    'm': 1650,
    'g': 9.81,
    'f0': 0.1,
    'f1': 5,
    'f2': 0.25,
    'ca': 0.3,
    'cd': 0.3,
    'T': 1.8,
    'u_max': 0.3 * 1650 * 9.81,
    'u_min': -0.3 * 1650 * 9.81,
    'clf': {'rate': 5},
    'cbf': {'rate': 5},
    'weight': {
    'slack': 2e-2,
    'input': 2/1650**2
    }
}

acc_sys = ACC(params)

controller = acc_sys.ctrl_clf_qp

dt = 0.02
t=0
sim_t = 20
x0 = np.array([0, 20, 100])

total_k = int(np.ceil(sim_t / dt))

# Initialize traces
xs = np.zeros((total_k, acc_sys.xdim))
ts = np.zeros(total_k)
us = np.zeros((total_k - 1, 1))
Vs = np.zeros(total_k - 1)
slack = np.zeros(total_k - 1)
xs[0] = x0
ts[0] = 0

for k in range(total_k - 1):
    u, slack[k], V, feas = controller(xs[k])
    us[k] = u
    Vs[k] = V
    #evaluate dynamics after one time step
    sol = solve_ivp(lambda t, s: acc_sys.dynamics(t, s, u), [ts[k], ts[k] + dt], xs[k], t_eval=[ts[k] + dt])
    x = sol.y[:, -1]
    xs[k + 1] = x
    ts[k + 1] = sol.t[-1]
    t += dt

# Plotting
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(ts, xs[:, 0], label='x')
plt.plot(ts, xs[:, 1], label='v')
plt.plot(ts, xs[:, 2], label='d')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(ts[:-1], us, label='u')
plt.legend()
plt.show()
