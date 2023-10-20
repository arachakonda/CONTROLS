import numpy as np
import mujoco as mj

s = np.sin
c = np.cos

#initialize utility functions
def radial_pose(pos_init, rot_mat):
    res = np.zeros(3)    
    mj.mju_mulMatVec(res, rot_mat, pos_init)
    return res

def rotation_matrix(theta_y, phi_x):
    rot_mat = np.array([c(theta_y), s(theta_y)*s(phi_x), 
                 s(theta_y)*c(phi_x),0,c(phi_x),-s(phi_x),
                -s(theta_y),s(phi_x)*c(theta_y),c(theta_y)*c(phi_x)], dtype=float)
    rot_mat = rot_mat.reshape(3,3)
    #print(rot_mat.shape)
    #print(rot_mat)
    return rot_mat

def rot2Quat(rot_mat):
    quat = np.zeros(4)
    rot_mat = rot_mat.reshape(9,1)
    mj.mju_mat2Quat(quat,rot_mat)
    return quat

def p2OLen(p,O):
    return np.linalg.norm(p-O) 

def rotMotMat(mot_or,theta):
    rot_mat = np.array([c(theta), 0, s(theta), 0, 1, 0, -s(theta), 0, c(theta) ])
    rot_mat = rot_mat.reshape(3,3)
    fin_mat = np.zeros((3,3))
    mj.mju_mulMatMat(fin_mat,mot_or,rot_mat)
    return fin_mat