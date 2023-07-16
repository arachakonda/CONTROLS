import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ipywidgets import interact
from IPython.display import HTML

class DiscreteTimeDynamics:

    def __init__(self, state_dim, control_dim, time_step):
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.time_step = time_step

    def plot_trajectory(self, x, u, figsize=(10, 10), fig=None, gs=None, return_ax=False):
        if fig is None:
            fig = plt.figure(figsize=figsize)
        if gs is None:
            gs = gridspec.GridSpec(2, 1)

        state_ax = fig.add_subplot(gs[0, 0])
        state_ax.set_ylabel('States')
        state_ax.plot(self.time_step * np.arange(len(x)), x)
        state_ax.legend(self.state_names, fontsize=24)

        control_ax = fig.add_subplot(gs[1, 0])
        control_ax.set_xlabel('Time')
        control_ax.set_ylabel('Controls')
        control_ax.set_prop_cycle(color=plt.rcParams['axes.prop_cycle'].by_key()['color'][self.state_dim:])
        control_ax.plot(self.time_step * np.arange(len(u)), u)
        control_ax.legend(self.control_names)

        return (fig, state_ax, control_ax) if return_ax else fig

    def animate_trajectory(self, x, u, figsize=(15, 8)):
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 2)

        _, state_ax, control_ax = self.plot_trajectory(x, u, fig=fig, gs=gs, return_ax=True)

        state_dots = [state_ax.scatter([0], x[0, i]) for i in range(self.state_dim)]
        control_dots = [control_ax.scatter([0], u[0, i]) for i in range(self.control_dim)]
        visualization_ax = fig.add_subplot(gs[:, 1])
        visualization_ax.tick_params(axis='both', labelsize=24)

        updateables = state_dots + control_dots + self._visualize(visualization_ax, x, u)

        def animate(t):
            for i, s in enumerate(state_dots):
                s.set_offsets([[self.time_step * t, x[t, i]]])
            for i, s in enumerate(control_dots):
                s.set_offsets([[self.time_step * t, u[t, i]]])
            self._update_visualization(x[t], u[t])
            return updateables

        plt.close()
        return HTML(
            animation.FuncAnimation(fig, animate, interval=1000 * self.time_step, blit=True,
                                    frames=len(u)).to_html5_video())

    @property
    def state_names(self):
        return range(self.state_dim)

    @property
    def control_names(self):
        return range(self.control_dim)

    def _visualize(self, ax, x, u):
        ax.text(0.2, 0.5, "visualization not implemented", fontsize=16)
        return []

    def _update_visualization(self, x, u):
        pass


class DoubleIntegatorDynamics(DiscreteTimeDynamics):

    def __init__(self, spatial_dim, time_step=0.1):
        super().__init__(2 * spatial_dim, spatial_dim, time_step)

        self.spatial_dim = spatial_dim
        self.A = np.eye(2 * spatial_dim) + time_step * np.eye(2 * spatial_dim, k=spatial_dim)
        self.B = (time_step**2 / 2 * np.eye(2 * spatial_dim, spatial_dim) +
                  time_step * np.eye(2 * spatial_dim, spatial_dim, k=-spatial_dim))

    def __call__(self, x, u):
        return self.A @ x + self.B @ u

    @property
    def state_names(self):
        if self.spatial_dim == 1:
            return ['s', 'v']
        return [f's_{i}' for i in range(self.spatial_dim)] + [f'v_{i}' for i in range(self.spatial_dim)]

    @property
    def control_names(self):
        if self.spatial_dim == 1:
            return ['a']
        return [f'a_{i}' for i in range(self.spatial_dim)]

    def _visualize(self, ax, x, u):
        plot_range = np.concatenate(
            [x[:-1, :self.spatial_dim] + x[:-1, self.spatial_dim:], x[:-1, :self.spatial_dim] + u], 0)
        plot_min = np.min(plot_range, 0)
        plot_max = np.max(plot_range, 0)
        plot_margin = (plot_max - plot_min) * 0.1
        plot_min = plot_min - plot_margin
        plot_max = plot_max + plot_margin
        ax.set_xlim(plot_min[0], plot_max[0])
        if self.spatial_dim == 1:
            ax.set_ylim(-1, 1)
        else:
            ax.set_ylim(plot_min[1], plot_max[1])
        p, v, a = x[0, :self.spatial_dim], x[0, self.spatial_dim:], u[0]
        if self.spatial_dim == 1:
            p, v, a = np.pad(p, [0, 1]), np.pad(v, [0, 1]), np.pad(a, [0, 1])
        self.a_vector = plt.quiver(p[0],
                                   p[1],
                                   a[0],
                                   a[1],
                                   angles='xy',
                                   scale_units='xy',
                                   scale=1,
                                   width=0.02,
                                   color='red')
        self.v_vector = plt.quiver(p[0],
                                   p[1],
                                   v[0],
                                   v[1],
                                   angles='xy',
                                   scale_units='xy',
                                   scale=1,
                                   width=0.01,
                                   color='black')
        return [self.a_vector, self.v_vector]

    def _update_visualization(self, x, u):
        p, v, a = x[:self.spatial_dim], x[self.spatial_dim:], u
        if self.spatial_dim == 1:
            p, v, a = np.pad(p, [0, 1]), np.pad(v, [0, 1]), np.pad(a, [0, 1])
        self.a_vector.set_offsets([[p[0], p[1]]])
        self.v_vector.set_offsets([[p[0], p[1]]])
        self.a_vector.set_UVC(a[0], a[1])
        self.v_vector.set_UVC(v[0], v[1])


def rollout_with_feedback(dynamics, controller, x0, N):
    x = np.zeros((N + 1, dynamics.state_dim))
    u = np.zeros((N, dynamics.control_dim))

    x[0] = x0
    for t in range(N):
        u[t] = controller(x[t])
        x[t + 1] = dynamics(x[t], u[t])

    return x, u


def rollout_control_sequence(dynamics, control_sequence, x0):
    N = len(control_sequence)
    x = np.zeros((N + 1, dynamics.state_dim))

    x[0] = x0
    for t in range(N):
        x[t + 1] = dynamics(x[t], u[t])

    return x, u

# Problem setup
dynamics = DoubleIntegatorDynamics(spatial_dim=1, time_step=0.1)

time_horizon = 5
T = int(time_horizon / dynamics.time_step)
x0 = np.array([5, 0])

# Feedback control

def linear_feedback_controller(k_p, k_d):

    def controller(x):
        return -k_p * x[0] - k_d * x[1]

    return controller

x, u = rollout_with_feedback(dynamics, linear_feedback_controller(k_p=1.0, k_d=1.0), x0, T)
dynamics.animate_trajectory(x, u)