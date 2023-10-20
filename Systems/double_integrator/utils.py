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
