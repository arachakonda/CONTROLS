import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import time as tim

class Interactions:
    def __init__(self,model, data, cam, opt):
        # For callback functions
        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.lastx = 0
        self.lasty = 0
        self.model = model
        self.data = data
        self.cam = cam
        self.opt = opt

        # Init GLFW, create window, make OpenGL context current, request v-sync
        glfw.init()
        self.window = glfw.create_window(1200, 900, "Demo", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        # initialize visualization data structures
        mj.mjv_defaultCamera(self.cam)
        mj.mjv_defaultOption(self.opt)
        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)

        # install GLFW mouse and keyboard callbacks
        glfw.set_key_callback(self.window, self.keyboard)
        glfw.set_cursor_pos_callback(self.window, self.mouse_move)
        glfw.set_mouse_button_callback(self.window, self.mouse_button)
        glfw.set_scroll_callback(self.window, self.scroll)

        cam.azimuth = 168.79999999999887
        cam.elevation = -4.2000000000000375
        cam.distance =  6.90761036059209
        cam.lookat = np.array([ -0.09023562087362373 , -0.12170080422270708 , 0.8632756268734745 ])

        opt.frame = mj._enums.mjtFrame.mjFRAME_NONE
        opt.label = mj._enums.mjtLabel.mjLABEL_NONE
        option_flag = mj._enums.mjtVisFlag.mjVIS_COM
        opt.flags[option_flag] = 0

    def keyboard(self,window, key, scancode, act, mods):
        if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
            mj.mj_resetData(self.model, self.data)
            mj.mj_forward(self.model, self.data)

    def mouse_button(self,window, button, act, mods):
        self.button_left = (glfw.get_mouse_button(
            window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
        self.button_middle = (glfw.get_mouse_button(
            window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
        self.button_right = (glfw.get_mouse_button(
            window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)
        # update mouse position
        glfw.get_cursor_pos(window)

    def mouse_move(self,window, xpos, ypos):
        dx = xpos - self.lastx
        dy = ypos - self.lasty
        self.lastx = xpos
        self.lasty = ypos

        # no buttons down: nothing to do
        if (not self.button_left) and (not self.button_middle) and (not self.button_right):
            return

        # get current window size
        width, height = glfw.get_window_size(window)

        # get shift key state
        PRESS_LEFT_SHIFT = glfw.get_key(
            window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
        PRESS_RIGHT_SHIFT = glfw.get_key(
            window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

        # determine action based on mouse button
        if self.button_right:
            if mod_shift:
                action = mj.mjtMouse.mjMOUSE_MOVE_H
            else:
                action = mj.mjtMouse.mjMOUSE_MOVE_V
        elif self.button_left:
            if mod_shift:
                action = mj.mjtMouse.mjMOUSE_ROTATE_H
            else:
                action = mj.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mj.mjtMouse.mjMOUSE_ZOOM

        mj.mjv_moveCamera(self.model, action, dx/height,
                        dy/height, self.scene, self.cam)

    def scroll(self,window, xoffset, yoffset):
        action = mj.mjtMouse.mjMOUSE_ZOOM
        mj.mjv_moveCamera(self.model, action, 0.0, -0.05 *
                        yoffset, self.scene, self.cam)
        
    def window_closed(self,window):
        if glfw.window_should_close(window):
            return 1
        else:
            return 0
    def terminate(self,window):
        glfw.terminate()
    
    def gui_interactions(self,window):
            # get framebuffer viewport
        viewport_width, viewport_height = glfw.get_framebuffer_size(
            window)
        viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

        # Update scene and render
        mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                        mj.mjtCatBit.mjCAT_ALL.value, self.scene)
        mj.mjr_render(viewport, self.scene, self.context)

        # swap OpenGL buffers (blocking call due to v-sync)
        glfw.swap_buffers(self.window)

        # process pending GUI events, call GLFW callbacks
        glfw.poll_events()

    def print_camera_config(self,):
        #print camera configuration (help to initialize the view)
        print('cam.azimuth =',self.cam.azimuth,';','cam.elevation =',self.cam.elevation,';','cam.distance = ',self.cam.distance)
        print('cam.lookat =np.array([',self.cam.lookat[0],',',self.cam.lookat[1],',',self.cam.lookat[2],'])')

    def observation_minutes(self,n):
        t_end = tim.time() + 60 * n
        return t_end
    
    def timer(self,timer_time):
        if(tim.time() < timer_time):
            return 1
        else:
            return 0
        
    
