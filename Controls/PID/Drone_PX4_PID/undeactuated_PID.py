#!/usr/bin/env python
import sys
# ROS python API
import rospy
from sensor_msgs.msg import Imu
# 3D point & Stamped Pose msgs
from geometry_msgs.msg import Point, PoseStamped, TwistStamped
# import all mavros messages and services
from mavros_msgs.msg import *
from mavros_msgs.srv import *
from nav_msgs.msg import *
from trajectory_msgs.msg import MultiDOFJointTrajectory as Mdjt
# from msg_check.msg import PlotDataMsg
from scipy import linalg as la

from tf.transformations import euler_from_quaternion, quaternion_from_euler


from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import LinkStates
from gazebo_msgs.srv import GetModelState


import numpy as np
from tf.transformations import *
#import RPi.GPIO as GPIO
import time

# Flight modes class
# Flight modes are activated using ROS services

class fcuModes:
    def __init__(self):
        pass

    def setTakeoff(self):
        rospy.wait_for_service('mavros/cmd/takeoff')
        try:
            takeoffService = rospy.ServiceProxy('mavros/cmd/takeoff', mavros_msgs.srv.CommandTOL)
            takeoffService(altitude = 3)
        except rospy.ServiceException as e:
            print("Service takeoff call failed: ,%s")%e

    def setArm(self):
        rospy.wait_for_service('mavros/cmd/arming')
        try:
            armService = rospy.ServiceProxy('mavros/cmd/arming', mavros_msgs.srv.CommandBool)
            armService(True)
        except rospy.ServiceException as e:
            print("Service arming call failed: %s")%e

    def setDisarm(self):
        rospy.wait_for_service('mavros/cmd/arming')
        try:
            armService = rospy.ServiceProxy('mavros/cmd/arming', mavros_msgs.srv.CommandBool)
            armService(False)
        except rospy.ServiceException as e:
            print("Service disarming call failed: %s")%e

    def setStabilizedMode(self):
        rospy.wait_for_service('mavros/set_mode')
        try:
            flightModeService = rospy.ServiceProxy('mavros/set_mode', mavros_msgs.srv.SetMode)
            flightModeService(custom_mode='STABILIZED')
        except rospy.ServiceException as e:
            print("service set_mode call failed: %s. Stabilized Mode could not be set.")%e

    def setOffboardMode(self):
        rospy.wait_for_service('mavros/set_mode')
        try:
            flightModeService = rospy.ServiceProxy('mavros/set_mode', mavros_msgs.srv.SetMode)
            flightModeService(custom_mode='OFFBOARD')
        except rospy.ServiceException as e:
            print("service set_mode call failed: %s. Offboard Mode could not be set.")%e

    def setAltitudeMode(self):
        rospy.wait_for_service('mavros/set_mode')
        try:
            flightModeService = rospy.ServiceProxy('mavros/set_mode', mavros_msgs.srv.SetMode)
            flightModeService(custom_mode='ALTCTL')
        except rospy.ServiceException as e:
            print("service set_mode call failed: %s. Altitude Mode could not be set.")%e

    def setPositionMode(self):
        rospy.wait_for_service('mavros/set_mode')
        try:
            flightModeService = rospy.ServiceProxy('mavros/set_mode', mavros_msgs.srv.SetMode)
            flightModeService(custom_mode='POSCTL')
        except rospy.ServiceException as e:
            print("service set_mode call failed: %s. Position Mode could not be set.")%e

    def setAutoLandMode(self):
        rospy.wait_for_service('mavros/set_mode')
        try:
            flightModeService = rospy.ServiceProxy('mavros/set_mode', mavros_msgs.srv.SetMode)
            flightModeService(custom_mode='AUTO.LAND')
        except rospy.ServiceException as e:
               print("service set_mode call failed: %s. Autoland Mode could not be set.")%e


class Controller:
    # initialization method
    def __init__(self):
        # Drone state
        self.state = State()
        # Instantiate a setpoints message
        self.sp = PoseStamped()
        self.payload_state = PoseStamped()
        self.payload_vel  = TwistStamped()
        # set the flag to use position setpoints and yaw angle
       
        # Step size for position update
        self.STEP_SIZE = 2.0
        # Fence. We will assume a square fence for now
        self.FENCE_LIMIT = 5.0

        # A Message for the current local position of the drone

        # initial values for setpoints
        self.cur_pose = PoseStamped()
        self.cur_vel = TwistStamped()
        self.imu = Imu()
        self.sp.pose.position.x = 0.0
        self.sp.pose.position.y = 0.0
        self.ALT_SP = 1.0
        self.sp.pose.position.z = self.ALT_SP
        self.local_pos = Point(0.0, 0.0, self.ALT_SP)
        self.local_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.desVel = np.zeros(3)
        self.errInt_a = np.zeros(3)
        self.errInt_u = np.zeros(2)

        self.desAcc = np.zeros(3)
        self.rho =  0.0
        self.att_cmd = PoseStamped()
        self.thrust_cmd = Thrust()

        self.count = True

        # Control parameters
        self.M = 1.5
        self.m = 0.01
        self.l = 0.3

        self.curAccPrev = np.array([0.,0.,0.])
        self.thrust_prev =np.array([0.,0.,0.])

        self.KP_a = np.array([1.0, 1.0, 6.0])
        self.KD_a = np.array([0.1, 0.1, 4.0])
        self.KI_a = np.array([0.02, 0.02, 0.5])

        self.KP_u = np.array([[0.2, 0.0],[0.0, 0.2],[0.0,0.0]])
        self.KD_u = np.array([[0.02, 0],[0.0, 0.02],[0.000,0.000]])
        self.KI_u = np.array([[0.002, 0],[0.0, 0.002],[0.000,0.000]])

        self.max_acc = 14   # orginally 4
        self.norm_thrust_const = 0.06
        self.max_th = 12.0
        self.max_throttle = 0.98

        self.gravity = np.array([0, 0, 9.8])
        self.pre_time = rospy.get_time()    
        # self.data_out = PlotDataMsg()

        # Publishers
        self.att_pub = rospy.Publisher('mavros/setpoint_attitude/attitude', PoseStamped, queue_size=10)
        self.thrust_pub = rospy.Publisher('mavros/setpoint_attitude/thrust', Thrust, queue_size=10)
        # self.data_pub = rospy.Publisher('/data_out', PlotDataMsg, queue_size=10)
        self.armed = False
        self.pin_1 = 16
        self.pin_2 = 18
        # GPIO.setmode(GPIO.BOARD)
        # GPIO.setup(self.pin_1, GPIO.OUT)
        # GPIO.setup(self.pin_2, GPIO.OUT)


    def multiDoFCb(self, msg):
        pt = msg.points[0]
        self.sp.pose.position.x = pt.transforms[0].translation.x
        self.sp.pose.position.y = pt.transforms[0].translation.y
        self.sp.pose.position.z = pt.transforms[0].translation.z
        self.data_out.sq = self.sp.pose.position
        self.desVel = np.array([pt.velocities[0].linear.x, pt.velocities[0].linear.y, pt.velocities[0].linear.z])
        self.desAcc = np.array([pt.accelerations[0].linear.x, pt.accelerations[0].linear.y, pt.accelerations[0].linear.z])
        # self.desVel = np.array([pt.accelerations[0].linear.x, pt.accelerations[0].linear.y, pt.accelerations[0].linear.z])
        # self.array2Vector3(self.sp.pose.position, self.data_out.acceleration)

        if (pt.transforms[0].translation.x < 0.01 and pt.transforms[0].translation.x > -0.01) \
        and (pt.transforms[0].translation.y > -1.01 and pt.transforms[0].translation.y < -0.99 ):            
            # GPIO.output(self.pin_1, True)
            print("Dropping first payload")
            time.sleep(0.1)
            # self.switch = 2.0
            # print(self.switch)
        
        elif (pt.transforms[0].translation.x < 0.01 and pt.transforms[0].translation.x > -0.01) \
        and (pt.transforms[0].translation.y < 1.01 and pt.transforms[0].translation.y > 0.99 ):
            # GPIO.output(self.pin_2, True)frho
            print("Dropping second payload")
            time.sleep(0.1)
            # self.switch = 1.0
            # print(self.switch)

    ## local position callback
    def posCb(self, msg):
        self.local_pos.x = msg.pose.position.x
        self.local_pos.y = msg.pose.position.y
        self.local_pos.z = msg.pose.position.z
        self.local_quat[0] = msg.pose.orientation.x
        self.local_quat[1] = msg.pose.orientation.y
        self.local_quat[2] = msg.pose.orientation.z
        self.local_quat[3] = msg.pose.orientation.w

    ## Drone State callback
    def stateCb(self, msg):
        self.state = msg

    ## Update setpoint message
    def updateSp(self):
        self.sp.pose.position.x = self.local_pos.x
        self.sp.pose.position.y = self.local_pos.y
        # self.sp.position.z = self.local_pos.z

    def odomCb(self, msg):
        self.cur_pose.pose.position.x = msg.pose.pose.position.x
        self.cur_pose.pose.position.y = msg.pose.pose.position.y
        self.cur_pose.pose.position.z = msg.pose.pose.position.z

        self.cur_pose.pose.orientation.w = msg.pose.pose.orientation.w
        self.cur_pose.pose.orientation.x = msg.pose.pose.orientation.x
        self.cur_pose.pose.orientation.y = msg.pose.pose.orientation.y
        self.cur_pose.pose.orientation.z = msg.pose.pose.orientation.z

        self.cur_vel.twist.linear.x = msg.twist.twist.linear.x
        self.cur_vel.twist.linear.y = msg.twist.twist.linear.y
        self.cur_vel.twist.linear.z = msg.twist.twist.linear.z

        self.cur_vel.twist.angular.x = msg.twist.twist.angular.x
        self.cur_vel.twist.angular.y = msg.twist.twist.angular.y
        self.cur_vel.twist.angular.z = msg.twist.twist.angular.z

    def accCB(self,msg):
        self.imu.orientation.w = msg.orientation.w
        self.imu.orientation.x = msg.orientation.x
        self.imu.orientation.y = msg.orientation.y
        self.imu.orientation.z = msg.orientation.z

        self.imu.angular_velocity.x = msg.angular_velocity.x
        self.imu.angular_velocity.y = msg.angular_velocity.y
        self.imu.angular_velocity.z = msg.angular_velocity.z

        self.imu.linear_acceleration.x = msg.linear_acceleration.x
        self.imu.linear_acceleration.y = msg.linear_acceleration.y
        self.imu.linear_acceleration.z = msg.linear_acceleration.z

    def newPoseCB(self, msg):
        if(self.sp.pose.position != msg.pose.position):
            print("New pose received")
        self.sp.pose.position.x = msg.pose.position.x
        self.sp.pose.position.y = msg.pose.position.y
        self.sp.pose.position.z = msg.pose.position.z
   
        self.sp.pose.orientation.x = msg.pose.orientation.x
        self.sp.pose.orientation.y = msg.pose.orientation.y
        self.sp.pose.orientation.z = msg.pose.orientation.z
        self.sp.pose.orientation.w = msg.pose.orientation.w

    def vector2Arrays(self, vector):        
        return np.array([vector.x, vector.y, vector.z])

    def vector3Arrays(self, vector):        
        return np.array([vector.x, vector.y, vector.z , vector.w])

    def array2Vector3(self, array, vector):
        vector.x = array[0]
        vector.y = array[1]
        vector.z = array[2]

    def array2Vector4(self, array, vector):
        vector.x = array[0]
        vector.y = array[1]
        vector.z = array[2]
        vector.w = array[3]

    def sigmoid(self, s, v):
        if np.absolute(s) > v:
            return s/np.absolute(s)
        else:
            return s/v

    def payloadCb(self, msg):
        self.payload_pos.x = msg.pose.position.x
        self.payload_pos.y = msg.pose.position.y
        self.payload_pos.z = msg.pose.position.z

    def payload_states(self, msg):
        idx = msg.name.index('iris::sphere')
        self.payload_state = msg.pose[idx]
        self.payload_vel = msg.twist[idx]

        x = np.round(-np.round(self.cur_pose.pose.position.x, 2) + np.round(self.payload_state.position.x, 2), 2)
        y = np.round(-np.round(self.cur_pose.pose.position.y, 2) + np.round(self.payload_state.position.y, 2), 2)
        z = np.round(-np.round(self.cur_pose.pose.position.z, 2) + np.round(self.payload_state.position.z, 2), 2)

        # self.alpha_ang = np.rad2deg(np.arctan2(x, z)) #theta
        # self.beta_ang = np.rad2deg(np.arctan2(y, z))  #phi

        self.alpha_ang = (np.arctan2(x, z)) #theta
        self.beta_ang = (np.arctan2(y, z))  #phi

    def a_des(self):
        dt = rospy.get_time() - self.pre_time
        self.pre_time = self.pre_time + dt
        if dt > 0.04:
            dt = 0.04
        if dt == 0.0:
            dt = 0.01

        curPos = self.vector2Arrays(self.cur_pose.pose.position)
        desPos = self.vector2Arrays(self.sp.pose.position)
        curVel = self.vector2Arrays(self.cur_vel.twist.linear)
        curAcc = self.vector2Arrays(self.imu.linear_acceleration)
        curor = self.vector3Arrays(self.cur_pose.pose.orientation)

        errPos_a = curPos - desPos
        errVel_a = curVel - self.desVel
        self.errInt_a += errPos_a*dt
        
        des_alpha = 0.0
        des_beta = 0.0
        alpha = self.alpha_ang   # Fx
        beta = self.beta_ang    # Fy

        err_alpha = alpha - des_alpha
        err_beta = beta - des_beta

        errPos_u = np.array([err_alpha, err_beta])
        errVel_u = np.array([err_alpha/dt, err_beta/dt])
        self.errInt_u += errPos_u*dt

        des_a =  (self.desAcc 
                  - np.multiply(self.KD_a, errVel_a) 
                  - np.multiply(self.KP_a, errPos_a) 
                  - np.multiply(self.KI_a, self.errInt_a) 
                  - np.dot(self.KD_u, errVel_u) 
                  - np.dot(self.KP_u, errPos_u) 
                  - np.dot(self.KI_u, self.errInt_u) 
                  + self.gravity)

        if np.linalg.norm(des_a) > self.max_acc:
            des_a = (self.max_acc/np.linalg.norm(des_a))*des_a

        print(des_a)

        return des_a

    def acc2quat(self,des_a, des_yaw):
        proj_xb_des = np.array([np.cos(des_yaw), np.sin(des_yaw), 0.0])
        if np.linalg.norm(des_a) == 0.0:
            zb_des = np.array([0,0,1])
        else:    
            zb_des = des_a / np.linalg.norm(des_a)
        yb_des = np.cross(zb_des, proj_xb_des) / np.linalg.norm(np.cross(zb_des, proj_xb_des))
        xb_des = np.cross(yb_des, zb_des) / np.linalg.norm(np.cross(yb_des, zb_des))
       
        rotmat = np.transpose(np.array([xb_des, yb_des, zb_des]))
        return rotmat

    def geo_con(self):
        des_a = self.a_des()    
        r_des = self.acc2quat(des_a, 0.0)
        rot_44 = np.vstack((np.hstack((r_des,np.array([[0,0,0]]).T)), np.array([[0,0,0,1]])))

        quat_des = quaternion_from_matrix(rot_44)
       
        zb = r_des[:,2]
        thrust = self.norm_thrust_const * des_a.dot(zb)
        # self.data_out.thrust = thrust
        thrust = np.maximum(0.0, np.minimum(thrust, 0.8))

        now = rospy.Time.now()
        self.att_cmd.header.stamp = now
        self.thrust_cmd.header.stamp = now
        # self.data_out.header.stamp = now
        self.att_cmd.pose.orientation.x = quat_des[0]
        self.att_cmd.pose.orientation.y = quat_des[1]
        self.att_cmd.pose.orientation.z = quat_des[2]
        self.att_cmd.pose.orientation.w = quat_des[3]
        self.thrust_cmd.thrust = thrust
        # self.data_out.orientation = self.att_cmd.pose.orientation

    def pub_att(self):
        self.geo_con()
        self.thrust_pub.publish(self.thrust_cmd)
        self.att_pub.publish(self.att_cmd)
        # self.data_pub.publish(self.data_out)


def main(argv):
   
    rospy.init_node('setpoint_node', anonymous=True)
    modes = fcuModes()  #flight modes
    cnt = Controller()  # controller object
    rate = rospy.Rate(30)
    rospy.Subscriber('mavros/state', State, cnt.stateCb)
    rospy.Subscriber('mavros/local_position/odom', Odometry, cnt.odomCb)
    rospy.Subscriber('mavros/imu/data', Imu, cnt.accCB)
    # Subscribe to payload position
    rospy.Subscriber('/gazebo/link_states', LinkStates, cnt.payload_states)

    # Subscribe to drone's local position
    rospy.Subscriber('mavros/local_position/pose', PoseStamped, cnt.posCb)
    rospy.Subscriber('new_pose', PoseStamped, cnt.newPoseCB)
    rospy.Subscriber('command/trajectory', Mdjt, cnt.multiDoFCb)
    sp_pub = rospy.Publisher('mavros/setpoint_position/local', PoseStamped, queue_size=10)

    print("ARMING")
    while not cnt.state.armed:
        modes.setArm()
        cnt.armed = True
        rate.sleep()

    cnt.armed = True
    k=0
    while k<20:
        sp_pub.publish(cnt.sp)
        rate.sleep()
        k = k + 1

    modes.setOffboardMode()
    print("---------")
    print("OFFBOARD")
    print("---------")

    # ROS main loop
    while not rospy.is_shutdown():
        # r_des = quaternion_matrix(des_orientation)[:3,:3]
        # r_cur = quaternion_matrix(cnt.local_quat)[:3,:3]

#--------------------------------------------
        cnt.pub_att()
        rate.sleep()
       

#--------------------------------------------  

if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except rospy.ROSInterruptException:
        pass
