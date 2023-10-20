import numpy as np

#task space trajectory generation and euler angle generation

#trajectory for circumduction

#because the workspace of the robot is a section of the hemisphere it should be able to execute any trajectory on that hemisphere


def eulThPhi(xyz,r):
    eul=np.zeros((xyz.shape[0],2))
    theta=np.arctan2(xyz[:,0],xyz[:,2])
    phi=np.arcsin(-xyz[:,1]/(r))
    eul[:,0] = theta
    eul[:,1] = phi
    return eul

def level_set(r,z,N):
    k= np.sqrt(r**2 - z**2)
    rad = np.linspace(0,2*np.pi,N)
    batch_aclock = np.zeros((N,3))
    batch_aclock[:,0]= k*np.cos(rad)
    batch_aclock[:,1]= k*np.sin(rad)
    batch_aclock[:,2]= z*np.ones(N)
    return batch_aclock


#print(eulThPhi(level_set(0.042,0.03,100),0.042))