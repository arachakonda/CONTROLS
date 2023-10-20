from control import Controller
from sims import SimUtils
from gui import Interactions

from utils import *

import numpy as np
import mujoco as mj

import pandas as pd

import sys


simend = 100

sim = SimUtils('pendulum.xml',simend)
interact = Interactions(sim.model, sim.data, sim.cam, sim.opt)

#controller code
control = Controller(sim)

n=len(sys.argv)
#print(n)
if(n>1):
    pass
else:
    pass


sim.data.qpos[0] = 0.01

#execute until the window is closed
mj.set_mjcb_control(control.energy_shaping_controller)

while not interact.window_closed(interact.window):

    time_prev = sim.data.time
    while (sim.data.time - time_prev <= 1/60):
        mj.mj_step(sim.model, sim.data)
    if(sim.data.time > simend):
        break

    interact.gui_interactions(interact.window)
    #interact.print_camera_config()

timer_time = interact.observation_minutes(1)

while interact.timer(timer_time):
    print("interaction  time")
    interact.gui_interactions(interact.window)
    #interact.print_camera_config()
    if interact.window_closed(interact.window):
        break

interact.terminate(interact.window)


