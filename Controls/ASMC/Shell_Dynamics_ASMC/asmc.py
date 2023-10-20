import mujoco as mj
from mujoco.glfw import glfw
import matplotlib.pyplot as plt
import numpy as np
import os
from traj import *
from paths import *
import pandas as pd

full_path = os.path.realpath(__file__)
path = os.path.dirname(full_path)
xml_path = '/'+path[1:]+'/' +'Socket_Assem_MJCF.xml' #xml file (assumes this is in the same folder as this file)
simend = 5 #simulation time
print_camera_config = 0 #set to 1 to print camera config
                        #this is useful for initializing view of the model)

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

t = []
qo0 = []
qr0 = []
qdo0 = []
qdr0 = []
qo1 = []
qr1 = []
qdo1 = []
qdr1 = []

previous_time = 0
scaling_factor =  1
k0_hat = 1e-2
k1_hat = 1e-2
k2_hat = 1e-2
warm_start_time = 1


def init_controller(model,data):
    #initialize the controller here. This function is called once, in the beginning
    #generate trajectory
    # global eul
    # eul = eulThPhi(level_set(0.020682,0.018,100),0.020682)
    # dict = {'theta': eul[:,0], 'psi': eul[:,1]}
    # df = pd.DataFrame(dict)
    # df.to_csv('waypoints.csv', index=False, header=True)

    global q_d, qd_d, qdd_d

    df = pd.read_csv('trajectories.csv', header=None, index_col=False)
    q_d = df.iloc[:,:2].to_numpy()
    qd_d = df.iloc[:,2:4].to_numpy()
    qdd_d = df.iloc[:,4:].to_numpy()

def save_data(t,qo0,qr0,qo1,qr1,qdo0,qdr0,qdo1,qdr1):
    dict ={'t': t, 'q_o0': qo0, 'q_r0':qr0, 'q_o1': qo1, 'q_r1': qr1,
           'qd_o0': qdo0, 'qd_r0':qdr0, 'qd_o1': qdo1, 'qd_r1': qdr1}
    df = pd.DataFrame(dict)
    df.to_csv('data.csv', index=False, header=True)


def controller(model, data):
    #put the controller here. This function is called inside the simulation.

    global q_d, qd_d, qdd_d, previous_time, k0_hat, k1_hat, k2_hat, warm_start_time

    time = data.time
    i= int(time/0.005)
    print("instance: %d"%i)
    if (time>=simend):
        time = simend

    # if (time<t_init):
    #     time = t_init

    #theta for shaft and psi for small shell
    # pd_0 = -kp*(data.qpos[0]-q_d[i][1])-kd*(data.qvel[0]-qd_d[i][1]) #the small shell trajectory control
    # pd_1 = -kp*(data.qpos[1]-q_d[i][0])-kd*(data.qvel[1]-qd_d[i][0]) #the shaft trajectory control
    time_step = data.time - previous_time
    print(time_step)
    if(time_step >= 0.001):
        if(time< warm_start_time):
            M = np.zeros((2,2))
            mj.mj_fullM(model,M,data.qM)
            f0 = data.qfrc_bias[0] # the small shell bias
            f1 = data.qfrc_bias[1] # the shaft bias
            f = np.array([f0,f1])

            kp = 500
            kd = 10*np.sqrt(kp)
            #theta for shaft and psi for small shell
            # pd_0 = -kp*(data.qpos[0]-q_d[i][1])-kd*(data.qvel[0]-qd_d[i][1]) #the small shell trajectory control
            # pd_1 = -kp*(data.qpos[1]-q_d[i][0])-kd*(data.qvel[1]-qd_d[i][0]) #the shaft trajectory control

            pd_0 = -kp*(data.sensordata[0]-q_d[i][1])-kd*(data.sensordata[1]-qd_d[i][1])#the small shell trajectory control
            pd_1 = -kp*(data.sensordata[2]-q_d[i][0])-kd*(data.sensordata[3]-qd_d[i][0])#the shaft trajectory control

            pd_control = np.array([pd_0,pd_1])
            tau_M_pd_control = np.matmul(M,pd_control)
            tau = np.add(tau_M_pd_control,f)
            data.ctrl[0] = tau[0]
            data.ctrl[1] = tau[1]

        else:
            pos_vec = np.array([data.sensordata[0],data.sensordata[2]])
            vel_vec = np.array([data.sensordata[1],data.sensordata[3]])
            # e_q = np.array(data.qpos) - np.array(np.flip(q_d[i]))
            # e_qd = np.array(data.qvel) - np.array(np.flip(qd_d[i]))

            e_q = pos_vec- np.array(np.flip(q_d[i]))
            e_qd = vel_vec - np.array(np.flip(qd_d[i]))

            # print(e_q)
            # print(e_qd)

            phi = np.zeros([model.nv,model.nv], dtype = float)
            phi[0,0] = 100
            phi[1,1] = 100

            phie_q = np.zeros(model.nv,dtype = float)
            mj.mju_mulMatVec(phie_q, phi, e_q)
            s = e_qd + phie_q

            zeta = np.array([e_q, e_qd])

            norm_zeta = np.linalg.norm(zeta)
            norm_e = np.linalg.norm(e_q)
            norm_ed = np.linalg.norm(e_qd)
            norm_s = np.linalg.norm(s)

            A = 1*np.identity(model.nv)
            A [0,0] = 0.0000000001 #equivalent to kp
            A [1,1] = 0.0000000001 #equivalent to kd

            gamma = np.exp(1)
            dol= gamma**(0.02*(norm_s+0.01))
            #dol = 1
            #print(dol)
            pol = time_step
            #pol = 1
            alpha = [(1/(dol*pol)),(1/(dol*pol)),(1/(dol*pol))]
            k0_hat = k0_hat + (scaling_factor*time_step*(norm_s*((norm_zeta)**(0)) - (alpha[0]*k0_hat)))
            k1_hat = k1_hat + (scaling_factor*time_step*(norm_s*((norm_zeta)**(1)) - (alpha[1]*k1_hat)))
            k2_hat = k2_hat + (scaling_factor*time_step*(norm_s*((norm_zeta)**(2)) - (alpha[2]*k2_hat)))

            print("time : %f"%(time))
            print("time step: %f"%(time_step))
            print("zeta norm: %f, e norm: %f, ed norm: %f, s norm: %f" %(norm_zeta,norm_e,norm_ed, norm_s))
            print("k0_hat %f, k1_hat: %f, k2_hat: %f"%(k0_hat,k1_hat,k2_hat))
            
            
            rho = (k0_hat) + (k1_hat*norm_zeta) + (k2_hat*((norm_zeta)**2))
            #rho = 0
            print("rho: %f"%(rho))
            print("")
            tau = -(np.matmul(A,s))-(rho*np.tanh(s))

            data.ctrl[0] = tau[0]
            data.ctrl[1] = tau[1]

        previous_time = data.time 

        
    

    t.append(data.time)
    # qo0.append(data.sensordata[0]) # small shell position
    qo0.append(data.qpos[0]) # small shell position
    qr0.append(q_d[i][1]) # desired position of small shell
    # qdo0.append(data.sensordata[1]) # small shell velocity
    qdo0.append(data.qvel[0]) # small shell velocity
    qdr0.append(qd_d[i][1]) # desired velocity of the small shell
    # print(data.qpos[0] - q_d[i][0])
    # qo1.append(data.sensordata[2]) # shaft position
    qo1.append(data.qpos[1]) # shaft position
    qr1.append(q_d[i][0]) # desired position of the shaft
    # qdo1.append(data.sensordata[3]) # shaft velocity
    qdo1.append(data.qvel[1]) # shaft velocity
    qdr1.append(qd_d[i][0]) # desired velocity of the shaft 
    # print(data.qpos[1] - q_d[i][1])
    # print("~~~~~~~~~")
    


def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
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
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)

#get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Example on how to set camera configuration
# cam.azimuth = 90
# cam.elevation = -45
# cam.distance = 2
# cam.lookat = np.array([0.0, 0.0, 0])
cam.azimuth = 142.99999999999986
cam.elevation = -12.800000000000034
cam.distance =  0.17442185354709724
cam.lookat =np.array([ 0.0005175632185703559 , -0.0012672639669149728 , 0.01610570946545535 ])


#initialize the controller
init_controller(model,data)
data.qpos[0] = q_d[0][1]
data.qpos[1] = q_d[0][0]

#set the controller
mj.set_mjcb_control(controller)
time_prev = data.time

while not glfw.window_should_close(window):

    # 5 seconds have been split into 1000 parts which means the trajectory is available at 200 points in 1 second
    # controller frequency can be greater than 200 Hz 
    # current time step =  0.002 => frequency = 500 Hz

    dt = data.time - time_prev
    mj.mj_step(model, data)
    while(1):
        dt = data.time - time_prev
        try:
            mj.mj_step(model, data)
            time_prev = data.time
        except:
            break
        
        if(dt> 1/1000):
            break
    if (data.time>=simend):
        save_data(t,qo0,qr0,qo1,qr1,qdo0,qdr0,qdo1,qdr1)
        break
    

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    #print camera configuration (help to initialize the view)
    if (print_camera_config==1):
        print('cam.azimuth =',cam.azimuth,';','cam.elevation =',cam.elevation,';','cam.distance = ',cam.distance)
        print('cam.lookat =np.array([',cam.lookat[0],',',cam.lookat[1],',',cam.lookat[2],'])')

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

    time_prev = data.time

glfw.terminate()
