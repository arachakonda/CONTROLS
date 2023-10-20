from asyncio.base_subprocess import ReadSubprocessPipeProto
import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import scipy
import os
import csv
xml_path = 'SARHARC.xml' #xml file (assumes this is in the same folder as this file)
simend = 20 #simulation time
print_camera_config = 0 #set to 1 to print camera config
                        #this is useful for initializing view of the model)

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0
datafile = "data.csv"
loop_index = 0
data_frequency = 10
previous_time =     
position_history = 0
ctrl_update_freq = 100
last_update = 0.0
warm_start_time = 1
scaling_factor =  1e-9
ctrl=0
q_ref = []
qd_ref = []
qdd_ref = []


print_time = 0

k0_hat = 1e-12
k1_hat = 1e-12
k2_hat = 1e-12


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
def init_controller(model,data):
    #initialize the controller here. This function is called once, in the beginning
    pass

def controller(model, data):
    #put the controller here. This function is called inside the simulation.
    pass

def init_save_data():
    global datafile
    fid = open(datafile, "w")
    fid.write("t, ")
    fid.write("mcp_t, ")
    fid.write("pip_t, ")
    fid.write("dip_t, ")
    fid.write("mcp_q, ")
    fid.write("pip_q, ")
    fid.write("dip_q, ")
    fid.write("mcp_qd, ")
    fid.write("pip_qd, ")
    fid.write("dip_qd, ")
    fid.write("mcp_qdd, ")
    fid.write("pip_qdd, ")
    fid.write("dip_qdd")
    #Don't remove the newline
    fid.write("\n")
    #get the full path
    fid.close()
def save_data(model, data):
    global datafile
    fid = open(datafile, "a")
    fid.write(str(data.time)+", ")
    fid.write(str(data.ctrl[4])+", ")
    fid.write(str(data.ctrl[5])+", ")
    fid.write(str(data.ctrl[6])+", ")
    fid.write(str(data.qpos[0])+", ")
    fid.write(str(data.qpos[1])+", ")
    fid.write(str(data.qpos[2])+", ")
    fid.write(str(data.qvel[0])+", ")
    fid.write(str(data.qvel[1])+", ")
    fid.write(str(data.qvel[2])+", ")
    fid.write(str(data.qacc[0])+", ")
    fid.write(str(data.qacc[1])+", ")
    fid.write(str(data.qacc[2])+", ")
    #Don't remove the newline
    fid.write("\n")
    fid.close()

def set_torque_control(model,actuator_no,flag):
    if flag ==0:
        model.actuator_gainprm[10*actuator_no+0] = 0
    else:
        model.actuator_gainprm[10*actuator_no+0] = 1

def set_position_control(model,actuator_no,kp):
        model.actuator_gainprm[10*actuator_no+0] = kp
        model.actuator_gainprm[10*actuator_no+1] = kp    

def set_position_control(model,actuator_no,kv):
        model.actuator_gainprm[10*actuator_no+0] = kv
        model.actuator_gainprm[10*actuator_no+1] = kv

def read_traj():
    fp_q_ref = open("q_ref.csv", "r")
    if not fp_q_ref:
        print("Can't open file")
    else:
        row = 0
        readcsv = csv.reader(fp_q_ref)
        for line in readcsv:
            column = []
            for col in range(len(line)):
                column.append(float(line[col]))
            q_ref.append(column)
    fp_q_ref.close()
    fp_qd_ref = open("qd_ref.csv", "r")
    if not fp_qd_ref:
        print("Can't open file")
    else:
        readcsv = csv.reader(fp_qd_ref)
        for line in readcsv:
            column = []
            for col in range(len(line)):
                column.append(float(line[col]))
            qd_ref.append(column)
    fp_qd_ref.close()
    fp_qdd_ref = open("qd_ref.csv", "r")
    if not fp_qdd_ref:
        print("Can't open file")
    else:
        row = 0
        readcsv = csv.reader(fp_qdd_ref)
        for line in readcsv:
            column = []
            for col in range(len(line)):
                column.append(float(line[col]))
            qdd_ref.append(column)
    fp_qdd_ref.close()

def init_controller(model,data):
    read_traj()

def mycontroller(model,data):
    global loop_index
    global data_frequency
    global print_time
    global k0_hat
    global k1_hat
    global k2_hat
    global previous_time
    t=0
    t_sec = 0
    t = data.time
    t_sec = int(t/0.5)
    #print(t_sec)
    tvec = [0,0,0]
    dense_M = np.zeros([model.nv,model.nv],dtype=float)
    mj.mj_fullM(model,dense_M,np.array(data.qM))

    e_q = np.array(data.qpos) - np.array(q_ref[t_sec])
    e_qd = np.array(data.qvel) - np.array(qd_ref[t_sec])
    e_qdd = np.array(data.qacc) -np.array(qdd_ref[t_sec])


    q = [data.qpos[0],data.qpos[1],data.qpos[2]]
    qd = [data.qvel[0], data.qvel[1], data.qvel[2]]
    qdd = [data.qacc[0],data.qacc[1],data.qacc[2]]
    
    time_step = data.time - previous_time


    if t_sec<= warm_start_time :
        if((time_step) > 0.001): 
            kp = 5
            kv = 5
            kde_qd = np.zeros(model.nv,dtype=float)
            kpe_q = np.zeros(model.nv,dtype=float)
            h= np.zeros(model.nv,dtype=float)
            mqdd= np.zeros(model.nv,dtype=float)
            mqdd_d= np.zeros(model.nv,dtype=float)
            mkde_qd =  np.zeros(model.nv,dtype=float)
            mkpe_q =  np.zeros(model.nv,dtype=float)
            qdd_d =  np.zeros(model.nv,dtype=float)


            q_init = [data.qpos[0],data.qpos[1],data.qpos[2]]
            qd_init = [0,0,0]
            qdd_init = [0,0,0]

            q_fin = [1.5708,1.309,1.22173]
            qd_fin = [0,0,0]
            qdd_fin = [0,0,0]

            qdd_d[0] = qdd_ref[t_sec][0]
            qdd_d[1] = qdd_ref[t_sec][1]
            qdd_d[2] = qdd_ref[t_sec][2]

            mj.mju_scl3(kde_qd, e_qd, kv)
            mj.mju_scl3(kpe_q, e_q, kp)
            mj.mju_mulMatVec(mqdd_d , dense_M, qdd_d)
            
            mj.mju_mulMatVec(mkde_qd, dense_M, kde_qd)
            mj.mju_mulMatVec(mkpe_q, dense_M, kpe_q)
            

            h[0] = data.qfrc_bias[0]
            h[1] = data.qfrc_bias[1]
            h[2] = data.qfrc_bias[2]
            
            c_mcp = h[0] + mqdd_d[0] - mkde_qd[0] - mkpe_q[0]
            c_pip = h[1] + mqdd_d[1] - mkde_qd[1] - mkpe_q[1]
            c_dip = h[2] + mqdd_d[2] - mkde_qd[2] - mkpe_q[2]
            
            tvec[0]=c_mcp
            tvec[1]=c_pip
            tvec[2]=c_dip
            
            data.ctrl[4] = tvec[0]
            data.ctrl[5] = tvec[1]
            data.ctrl[6] = tvec[2]


    if  t_sec > warm_start_time and t_sec<=40 :

        if((time_step)> 0.001):
            phi = np.zeros([model.nv,model.nv], dtype = float)
            phi[0,0] = 10
            phi[1,1] = 1
            phi[2,2] = 1

            phie_q = np.zeros(model.nv,dtype = float)
            mj.mju_mulMatVec(phie_q, phi, e_q)
            s = e_qd + phie_q

            zeta = np.array([e_q, e_qd])

            norm_zeta = np.linalg.norm(zeta)
            norm_e = np.linalg.norm(e_q)
            norm_ed = np.linalg.norm(e_qd)
            norm_s = np.linalg.norm(s)


            A = 1*np.identity(model.nv)
            A [0,0] = 0.01
            A [1,1] = 0.01
            A [2,2] = 0.001
            gamma = np.exp(1)
            dol= gamma**(0.02*(norm_s+0.01))
            print(dol)

            k0_hat = k0_hat + (scaling_factor*(np.linalg.norm(s)*((norm_zeta)**(0)) - (1/(dol*time_step)*k0_hat))*time_step)
            k1_hat = k1_hat + (scaling_factor*(np.linalg.norm(s)*((norm_zeta)**(1)) -  (1/(dol*time_step)*k1_hat))*time_step)
            k2_hat = k2_hat + (scaling_factor*(np.linalg.norm(s)*((norm_zeta)**(2)) - (1/(dol*time_step)*k2_hat))*time_step)

            rho = (k0_hat) + (k1_hat*norm_zeta) + (k2_hat*((norm_zeta)**2))
            #rho = 0

            tau = -(A.dot(s))-(rho*np.tanh(s))

            data.ctrl[4] = tau[0]
            data.ctrl[5] = tau[1]
            data.ctrl[6] = tau[2]
                

            if (print_time != t_sec):                

                #print(np.linalg.norm(s)*((norm_zeta)**(0)))

                print("time : %f"%(t_sec))
                print("time step: %f"%(time_step))
                print("tau:%s"%np.array2string(tau))
                print("rho: %f"%(rho))
                print("zeta norm: %f, e norm: %f, ed norm: %f, s norm: %f" %(norm_zeta,norm_e,norm_ed, norm_s))
                print("k0_hat %f, k1_hat: %f, k2_hat: %f"%(k0_hat,k1_hat,k2_hat))
                print("")
                print_time = t_sec
                R = np.array([[-0.004434, -0.00464658, -0.016865],[0.004434, 0.00464658, 0.016865],[0,0.00464658,0.016865], [0,0,0.016865]])
                tensions = np.linalg.pinv(R.T).dot(tau)+ scipy.linalg.null_space(R.T, rcond = 1)


    if loop_index%data_frequency==0:
        save_data(model,data)
        # print("here")

    


    loop_index = loop_index + 1
    previous_time = data.time 
          

dirname = os.path.dirname(os.path.abspath(__file__))
print(dirname)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = "SARHARC.xml"

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
data.qpos[0] = 0.3
data.qpos[1] = 0.3
data.qpos[2] = 0.3
# Example on how to set camera configuration
cam.azimuth = -104.449080
cam.elevation = -19.016950
cam.distance = 0.397198
cam.lookat = np.array([0.009315, -0.023338, -0.045315])
init_save_data()
#initialize the controller
init_controller(model,data)
#set the controller
mj.set_mjcb_control(mycontroller)
#-104.449080, -19.016950, 0.397198, 0.009315, -0.023338, -0.045315
while not glfw.window_should_close(window):
    time_prev = data.time

    while (data.time - time_prev < 1.0/60.0):
        mj.mj_step(model, data)

    if (data.time>=simend):
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

glfw.terminate()

