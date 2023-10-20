#!/usr/bin/env python
import sys
# ROS python API
import rospy

# 3D point & Stamped Pose msgs
from geometry_msgs.msg import Point, PoseStamped, TwistStamped
# import all mavros messages and services
from mavros_msgs.msg import *
from mavros_msgs.srv import *
from nav_msgs.msg import *
from trajectory_msgs.msg import MultiDOFJointTrajectory as Mdjt
from std_msgs.msg import Float64
# from msg_check.msg import PlotDataMsg
from msg_check.msg import SwDataMsg


import numpy as np
from tf.transformations import *
# import RPi.GPIO as GPIO
import time


# Kpos = np.array([-2, -2, -3])
# Kvel = np.array([-2, -2, -3])
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
        except rospy.ServiceException, e:
            print "Service takeoff call failed: %s"%e

    def setArm(self):
        rospy.wait_for_service('mavros/cmd/arming')
        try:
            armService = rospy.ServiceProxy('mavros/cmd/arming', mavros_msgs.srv.CommandBool)
            armService(True)
        except rospy.ServiceException, e:
            print "Service arming call failed: %s"%e

    def setDisarm(self):
        rospy.wait_for_service('mavros/cmd/arming')
        try:
            armService = rospy.ServiceProxy('mavros/cmd/arming', mavros_msgs.srv.CommandBool)
            armService(False)
        except rospy.ServiceException, e:
            print "Service disarming call failed: %s"%e

    def setStabilizedMode(self):
        rospy.wait_for_service('mavros/set_mode')
        try:
            flightModeService = rospy.ServiceProxy('mavros/set_mode', mavros_msgs.srv.SetMode)
            flightModeService(custom_mode='STABILIZED')
        except rospy.ServiceException, e:
            print "service set_mode call failed: %s. Stabilized Mode could not be set."%e

    def setOffboardMode(self):
        rospy.wait_for_service('mavros/set_mode')
        try:
            flightModeService = rospy.ServiceProxy('mavros/set_mode', mavros_msgs.srv.SetMode)
            flightModeService(custom_mode='OFFBOARD')
        except rospy.ServiceException, e:
            print "service set_mode call failed: %s. Offboard Mode could not be set."%e

    def setAltitudeMode(self):
        rospy.wait_for_service('mavros/set_mode')
        try:
            flightModeService = rospy.ServiceProxy('mavros/set_mode', mavros_msgs.srv.SetMode)
            flightModeService(custom_mode='ALTCTL')
        except rospy.ServiceException, e:
            print "service set_mode call failed: %s. Altitude Mode could not be set."%e

    def setPositionMode(self):
        rospy.wait_for_service('mavros/set_mode')
        try:
            flightModeService = rospy.ServiceProxy('mavros/set_mode', mavros_msgs.srv.SetMode)
            flightModeService(custom_mode='POSCTL')
        except rospy.ServiceException, e:
            print "service set_mode call failed: %s. Position Mode could not be set."%e

    def setAutoLandMode(self):
        rospy.wait_for_service('mavros/set_mode')
        try:
            flightModeService = rospy.ServiceProxy('mavros/set_mode', mavros_msgs.srv.SetMode)
            flightModeService(custom_mode='AUTO.LAND')
        except rospy.ServiceException, e:
               print "service set_mode call failed: %s. Autoland Mode could not be set."%e

class Controller:
    # initialization method
    def __init__(self):
        # Drone state
        self.state = State()
        # Instantiate a setpoints message
        self.sp = PoseStamped()
        # set the flag to use position setpoints and yaw angle
       
        # Step size for position update
        self.STEP_SIZE = 2.0
        # Fence. We will assume a square fence for now
        self.FENCE_LIMIT = 5.0

        # A Message for the current local position of the drone

        # initial values for setpoints
        self.cur_pose = PoseStamped()
        self.cur_vel = TwistStamped()
        self.sp.pose.position.x = 0.0
        self.sp.pose.position.y = 0.0
        self.ALT_SP = 1.0
        self.sp.pose.position.z = self.ALT_SP
        self.local_pos = Point(0.0, 0.0, self.ALT_SP)
        self.local_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.desVel = np.zeros(3)
        # self.errInt = np.zeros(3)
        self.att_cmd = PoseStamped()
        self.thrust_cmd = Thrust()

        self.switch = Float64()
        self.switch = 1.0


        #COMMON PARAMETERS
        self.norm_thrust_const = 0.06
        self.max_th = 12
        self.max_throttle = 0.99
        self.v = 0.1
        self.gravity = np.array([0, 0, 9.8])
        self.Phi = np.array([0.8, 0.8, 0.8])

        self.xi_p = np.array([1.0, 1.0, 1.0])


        # Control parameters switch 1 ORIGINAL
        self.Kp0_1 = np.array([1.0, 1.0, 1.0])
        self.Kp1_1 = np.array([2.0, 2.0, 2.0])
        self.Lam_1 = np.array([3.0, 6.0, 6.0])
        self.M_1 = 0.1
        self.alpha_m_1 = 0.02
        self.alpha0_1 = np.array([1.0,1.0,1.0])
        self.alpha1_1 = np.array([3.0,3.0,3.0])

        #adding rho1 parameters
        self.rho_p0_1 = np.array([0.01, 0.01, 0.01])
        self.rho_p1_1 = np.array([0.01, 0.01, 0.01])

        self.beta_p0_1 = np.array([2.0, 2.0, 2.0])
        self.beta_p1_1 = np.array([2.0, 2.0, 2.0])
        self.nu_p0_1 =  np.array([0.01, 0.01, 0.01])
        self.nu_p1_1 =  np.array([0.01, 0.01, 0.01])
        self.varrho_p_1 = np.minimum(np.min(self.Lam_1), np.min(self.Phi))/2.0
        

        # Control parameters switch 2 (MORE AGGRESIVE)
        self.Kp0_2 = np.array([1.0, 1.0, 1.0])
        self.Kp1_2 = np.array([2.0, 2.0, 2.0])
        self.Lam_2 = np.array([6.0, 8.0, 6.0])
        self.M_2 = 0.1
        self.alpha_m_2 = 0.01
        self.alpha0_2 = np.array([1.0,1.0,3.0])
        self.alpha1_2 = np.array([3.0,1.0,3.0])

        self.rho_p0_2 = np.array([0.01, 0.01, 0.01])
        self.rho_p1_2 = np.array([0.01, 0.01, 0.01])

        self.beta_p0_2 = np.array([2.0, 2.0, 2.0])
        self.beta_p1_2 = np.array([2.0, 2.0, 2.0])
        self.nu_p0_2 =  np.array([0.01, 0.01, 0.01])
        self.nu_p1_2 =  np.array([0.01, 0.01, 0.01])
        self.varrho_p_2 = np.minimum(np.min(self.Lam_2), np.min(self.Phi))/2.0


        # Control parameters switch 3
        self.Kp0_3 = np.array([1.0, 1.0, 1.0])
        self.Kp1_3 = np.array([2.0, 2.0, 2.0])
        self.Lam_3 = np.array([6.0, 8.0, 6.0])
        self.M_3 = 0.1
        self.alpha_m_3 = 0.01
        self.alpha0_3 = np.array([3.0,3.0,1.0])
        self.alpha1_3 = np.array([0.1,0.1,1.0])

        self.rho_p0_3 = np.array([0.01, 0.01, 0.01])
        self.rho_p1_3 = np.array([0.01, 0.01, 0.01])

        self.beta_p0_3 = np.array([2.0, 2.0, 2.0])
        self.beta_p1_3 = np.array([2.0, 2.0, 2.0])
        self.nu_p0_3 =  np.array([0.01, 0.01, 0.01])
        self.nu_p1_3 =  np.array([0.01, 0.01, 0.01])
        self.varrho_p_3 = np.minimum(np.min(self.Lam_3), np.min(self.Phi))/2.0


        self.pre_time = rospy.get_time()    

        # self.data_out = SwDataMsg()

        # Publishers
        self.att_pub = rospy.Publisher('mavros/setpoint_attitude/attitude', PoseStamped, queue_size=10)
        self.thrust_pub = rospy.Publisher('mavros/setpoint_attitude/thrust', Thrust, queue_size=10)
        self.sw_pub = rospy.Publisher('/switch', Float64, queue_size=10)
        # self.data_pub = rospy.Publisher('/data_out', PlotDataMsg, queue_size=10)
        # self.data_pub = rospy.Publisher('/data_out', SwDataMsg, queue_size=10)
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
        
        # self.data_out.sq = self.sp.pose.position
        # self.desVel = np.array([pt.velocities[0].linear.x, pt.velocities[0].linear.y, pt.velocities[0].linear.z])

        if (pt.transforms[0].translation.x < 0.503 and pt.transforms[0].translation.x > 0.497) \
        and (pt.transforms[0].translation.y > -0.497 and pt.transforms[0].translation.y < -0.503 ):            
            # GPIO.output(self.pin_1, True)
            print("Dropping first payload")
            time.sleep(0.1)
            self.switch = 2.0
            print(self.switch)
        
        elif (pt.transforms[0].translation.x < -0.498 and pt.transforms[0].translation.x > -0.502) \
        and (pt.transforms[0].translation.y < 0.502 and pt.transforms[0].translation.y > 0.498 ):
            # GPIO.output(self.pin_2, True)
            print("Dropping second payload")
            time.sleep(0.1)
            self.switch = 1.0
            print(self.switch)



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

    def switchState(self, msg):
        self.switch = msg.data
        print(self.switch)
        if self.switch==1.0:
            self.max_th = 10.0
        elif self.switch==2.0:
            self.max_th = 11.0
        elif self.switch==3.0:
            self.max_th = 12.0


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

    def array2Vector3(self, array, vector):
        vector.x = array[0]
        vector.y = array[1]
        vector.z = array[2]

    def sigmoid(self, s, v):
        if np.absolute(s) > v:
            return s/np.absolute(s)
        else:
            return s/v


    def th_des(self):
        dt = rospy.get_time() - self.pre_time
        self.pre_time = self.pre_time + dt
        if dt > 0.04:
            dt = 0.04

        curPos = self.vector2Arrays(self.cur_pose.pose.position)
        desPos = self.vector2Arrays(self.sp.pose.position)
        curVel = self.vector2Arrays(self.cur_vel.twist.linear)

        errPos = curPos - desPos
        self.xi_p = errPos
        errVel = curVel - self.desVel
        sv = errVel + np.multiply(self.Phi, errPos)

        # print(self.switch)


        if self.switch==1.0 :

            if self.armed:
                self.Kp0_1 += (sv - np.multiply(self.alpha0_1, self.Kp0_1))*dt
                self.Kp1_1 += (sv - np.multiply(self.alpha1_1, self.Kp1_1))*dt
                self.Kp0_1 = np.maximum(self.Kp0_1, 0.0001*np.ones(3))
                self.Kp1_1 = np.maximum(self.Kp1_1, 0.0001*np.ones(3))
                self.M_1 += (-sv[2] - self.alpha_m_1*self.M_1)*dt
                self.M_1 = np.maximum(self.M_1, 0.1)
                self.M_2 = np.maximum(self.M_1, 0.1)
                self.M_3 = np.maximum(self.M_1, 0.1)
                # print(self.M)

            Rho = self.Kp0_1 + self.rho_p0_1 + np.multiply((self.Kp1_1 + self.rho_p1_1), self.xi_p)

            delTau = np.zeros(3)
            delTau[0] = Rho[0]*self.sigmoid(sv[0],self.v)
            delTau[1] = Rho[1]*self.sigmoid(sv[1],self.v)
            delTau[2] = Rho[2]*self.sigmoid(sv[2],self.v)

            self.des_th = -np.multiply(self.Lam_1, sv) - errPos - delTau + self.M_1*self.gravity

            if np.linalg.norm(self.des_th) > self.max_th:
                self.des_th = (self.max_th/np.linalg.norm(self.des_th))*self.des_th

            # self.array2Vector3(self.Kp0_1, self.data_out.Kp0_1)
            # self.array2Vector3(self.Kp1_1, self.data_out.Kp1_1)
            # self.data_out.hatM_1 = self.M_1


        elif self.switch==2.0:

            if self.armed:
                self.Kp0_2 += (sv - np.multiply(self.alpha0_2, self.Kp0_2))*dt
                self.Kp1_2 += (sv - np.multiply(self.alpha1_2, self.Kp1_2))*dt
                self.Kp0_2 = np.maximum(self.Kp0_2, 0.0001*np.ones(3))
                self.Kp1_2 = np.maximum(self.Kp1_2, 0.0001*np.ones(3))
                self.M_2 += (-sv[2] - self.alpha_m_2*self.M_2)*dt
                self.M_2 = np.maximum(self.M_2, 0.1)
                self.M_3 = np.maximum(self.M_2, 0.1)

                # print(self.M)

            # Rho = self.Kp0_2 + self.Kp1_2*errPos
            Rho = self.Kp0_2 + self.rho_p0_2 + np.multiply((self.Kp1_2 + self.rho_p1_2), self.xi_p)

            delTau = np.zeros(3)
            delTau[0] = Rho[0]*self.sigmoid(sv[0],self.v)
            delTau[1] = Rho[1]*self.sigmoid(sv[1],self.v)
            delTau[2] = Rho[2]*self.sigmoid(sv[2],self.v)

            self.des_th = -np.multiply(self.Lam_2, sv) - errPos - delTau + self.M_2*self.gravity

            if np.linalg.norm(self.des_th) > self.max_th:
                self.des_th = (self.max_th/np.linalg.norm(self.des_th))*self.des_th

            # self.array2Vector3(self.Kp0_2, self.data_out.Kp0_2)
            # self.array2Vector3(self.Kp1_2, self.data_out.Kp1_2)
            # self.data_out.hatM_2 = self.M_2


        elif self.switch==3.0:

            if self.armed:
                self.Kp0_3 += (sv - np.multiply(self.alpha0_3, self.Kp0_3))*dt
                self.Kp1_3 += (sv - np.multiply(self.alpha1_3, self.Kp1_3))*dt
                self.Kp0_3 = np.maximum(self.Kp0_3, 0.0001*np.ones(3))
                self.Kp1_3 = np.maximum(self.Kp1_3, 0.0001*np.ones(3))
                self.M_3 += (-sv[2] - self.alpha_m_3*self.M_3)*dt
                self.M_3 = np.maximum(self.M_3, 0.1)
                # print(self.M)

            # Rho = self.Kp0_3 + self.Kp1_3*errPos
            Rho = self.Kp0_3 + self.rho_p0_3 + np.multiply((self.Kp1_3 + self.rho_p1_3), self.xi_p)

            delTau = np.zeros(3)
            delTau[0] = Rho[0]*self.sigmoid(sv[0],self.v)
            delTau[1] = Rho[1]*self.sigmoid(sv[1],self.v)
            delTau[2] = Rho[2]*self.sigmoid(sv[2],self.v)

            self.des_th = -np.multiply(self.Lam_3, sv) - errPos - delTau + self.M_3*self.gravity

            if np.linalg.norm(self.des_th) > self.max_th:
                self.des_th = (self.max_th/np.linalg.norm(self.des_th))*self.des_th

            # self.array2Vector3(self.Kp0_3, self.data_out.Kp0_3)
            # self.array2Vector3(self.Kp1_3, self.data_out.Kp1_3)
            # self.data_out.hatM_3 = self.M_3

        else:
            print("Inavlid")


        if(self.switch!=1):
            dot_rho_p0 = (-np.multiply((self.beta_p0_1 + 
                        np.multiply((self.varrho_p_1/2)*self.Kp0_1, self.Kp0_1)), self.rho_p0_1) +  
                        np.multiply(self.beta_p0_1, self.nu_p0_1))

            dot_rho_p1 = (-np.multiply((self.beta_p1_1 + 
                        np.multiply((self.varrho_p_1/2)*self.Kp1_1, self.Kp1_1)), self.rho_p1_1) + 
                        np.multiply(self.beta_p1_1, self.nu_p1_1))

            self.rho_p0_1 += dot_rho_p0*dt
            self.rho_p1_1 += dot_rho_p1*dt

            # self.array2Vector3(self.rho_p0_1, self.data_out.rho_p0_1)
            # self.array2Vector3(self.rho_p1_1, self.data_out.rho_p1_1)


        if(self.switch!=2):
            dot_rho_p0 = (-np.multiply((self.beta_p0_2 +
                        np.multiply((self.varrho_p_2/2)*self.Kp0_2, self.Kp0_2)), self.rho_p0_2) +
                        np.multiply(self.beta_p0_2, self.nu_p0_2))

            dot_rho_p1 = (-np.multiply((self.beta_p1_2 +
                        np.multiply((self.varrho_p_2/2)*self.Kp1_2, self.Kp1_2)), self.rho_p1_2) +
                        np.multiply(self.beta_p1_2, self.nu_p1_2))

            self.rho_p0_2 += dot_rho_p0*dt
            self.rho_p1_2 += dot_rho_p1*dt

            # self.array2Vector3(self.rho_p0_2, self.data_out.rho_p0_2)
            # self.array2Vector3(self.rho_p1_2, self.data_out.rho_p1_2)


        if(self.switch!=3):
            dot_rho_p0 = (-np.multiply((self.beta_p0_3 +
                        np.multiply((self.varrho_p_3/2)*self.Kp0_3, self.Kp0_3)), self.rho_p0_3)+
                        np.multiply(self.beta_p0_3, self.nu_p0_3))

            dot_rho_p1 = (-np.multiply((self.beta_p1_3 +
                        np.multiply((self.varrho_p_3/2)*self.Kp1_3, self.Kp1_3)), self.rho_p1_3)+
                        np.multiply(self.beta_p1_3, self.nu_p1_3))

            self.rho_p0_3 += dot_rho_p0*dt
            self.rho_p1_3 += dot_rho_p1*dt

            # self.array2Vector3(self.rho_p0_3, self.data_out.rho_p0_3)
            # self.array2Vector3(self.rho_p1_3, self.data_out.rho_p1_3)

        # self.array2Vector3(sv, self.data_out.sp)
        # self.array2Vector3(errPos, self.data_out.position_error)
        # self.array2Vector3(errVel, self.data_out.velocity_error)
        # self.array2Vector3(delTau, self.data_out.delTau_p)
        # self.array2Vector3(Rho, self.data_out.zeta_p)

        return self.des_th

    def acc2quat(self, des_th, des_yaw):
        proj_xb_des = np.array([np.cos(des_yaw), np.sin(des_yaw), 0.0])
        if np.linalg.norm(des_th) == 0.0:
            zb_des = np.array([0,0,1])
        else:    
            zb_des = des_th / np.linalg.norm(des_th)
        yb_des = np.cross(zb_des, proj_xb_des) / np.linalg.norm(np.cross(zb_des, proj_xb_des))
        xb_des = np.cross(yb_des, zb_des) / np.linalg.norm(np.cross(yb_des, zb_des))
       
        rotmat = np.transpose(np.array([xb_des, yb_des, zb_des]))
        return rotmat

    def geo_con(self):
        des_th = self.th_des()    
        r_des = self.acc2quat(des_th, 0.0)
        rot_44 = np.vstack((np.hstack((r_des,np.array([[0,0,0]]).T)), np.array([[0,0,0,1]])))

        quat_des = quaternion_from_matrix(rot_44)
       
        zb = r_des[:,2]
        thrust = self.norm_thrust_const * des_th.dot(zb)
        # self.data_out.thrust = thrust
        
        thrust = np.maximum(0.0, np.minimum(thrust, self.max_throttle))

        now = rospy.Time.now()
        self.att_cmd.header.stamp = now
        self.thrust_cmd.header.stamp = now
        # self.data_out.header.stamp = now
        self.att_cmd.pose.orientation.x = quat_des[0]
        self.att_cmd.pose.orientation.y = quat_des[1]
        self.att_cmd.pose.orientation.z = quat_des[2]
        self.att_cmd.pose.orientation.w = quat_des[3]
        self.thrust_cmd.thrust = thrust
        # print(thrust)
        # print(quat_des)

    def pub_att(self):
        self.geo_con()
        self.thrust_pub.publish(self.thrust_cmd)
        self.att_pub.publish(self.att_cmd)
        # self.data_pub.publish(self.data_out)

# Main function
def main(argv):
   
    rospy.init_node('setpoint_node', anonymous=True)
    modes = fcuModes()  #flight modes
    cnt = Controller()  # controller object
    rate = rospy.Rate(30)
    rospy.Subscriber('mavros/state', State, cnt.stateCb)

    rospy.Subscriber('mavros/local_position/odom', Odometry, cnt.odomCb)

    # Subscribe to drone's local position
    rospy.Subscriber('mavros/local_position/pose', PoseStamped, cnt.posCb)
    rospy.Subscriber('new_pose', PoseStamped, cnt.newPoseCB)
    rospy.Subscriber('command/trajectory', Mdjt, cnt.multiDoFCb)
    rospy.Subscriber('switch', Float64, cnt.switchState)
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
