"""
"""
from .libs.algorithms import RMHC_SLAM, Laser
from collections    import deque
import numpy as np
import cv2
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import settings

if settings.draw_dist:
    plt.ion()
    plt.figure()


class Pioneer(object):
    """
    Methods for controlling Pioneer robot in V-REP, that has a SICK_TiM310 sensor attached
    """
    def __init__(self, env):
        self.env = env
        
        # Lidar
        self.lidar_data     = []
        self._lidar_names   = ['SICK_TiM310_sensor1', 'SICK_TiM310_sensor2']
        self._lidar_handles = [self.env.get_handle(x) for x in self._lidar_names]   
        self._lidar_angle   = settings.lidar_angle
        self._lidar         = None        
        self._slam_engine   = None

        # Motors, positions and angles
        self._motor_names   = ['Pioneer_p3dx_leftMotor', 'Pioneer_p3dx_rightMotor']
        self._motor_handles = [self.env.get_handle(x) for x in self._motor_names]
        self.angular_velocity = np.zeros(2)
        self.angles = np.zeros(2)
        self.theta  = 0
        self.pos    = [0, 0]
        self.position_history = deque(maxlen=settings.position_history_length)
        self.change_velocity([0, 0])
        self.start_streaming()

    def init_lidar(self):
        self._lidar = Laser(
            len(self.lidar_data), 
            settings.scan_rate, 
            self._lidar_angle,
            settings.distance_no_detection_mm,
            settings.detection_margin,
            settings.offset_mm)
        self._slam_engine = RMHC_SLAM(self._lidar, settings.image_size, settings.map_size)

    def start_streaming(self):
        """
        Start streaming the _lidar data and joint angles to reduce overhead
        """
        stream1 = [self.env.read_lidar(x, 'streaming') for x in self._lidar_handles]
        stream2 = [self.env.get_joint_angle(x, 'streaming') for x in self._motor_handles]
        stream3 = [self.env.get_joint_velocity(x, 'streaming') for x in self._motor_handles]
        
    def current_speed(self):
        """
        Current angular velocity of the wheel motors in rad/s
        """
        prev_angles = np.copy(self.angles)
        self.angles = np.array([self.env.get_joint_angle(x, 'buffer') for x in self._motor_handles])
        angular_velocity = self.angles - prev_angles
        for i, v in enumerate(angular_velocity):
            # In case radians reset to 0
            if v < -np.pi:
                angular_velocity[i] =  np.pi*2 + angular_velocity[i]
            if v > np.pi:
                angular_velocity[i] = -np.pi*2 + angular_velocity[i]
        self.angular_velocity = angular_velocity
        return self.angular_velocity

    def current_speed_API(self):
        self.angular_velocity = np.array([self.env.get_joint_velocity(x, 'buffer') for x in self._motor_handles])
        return self.angular_velocity

    def change_velocity(self, velocities, target=None):
        """
        Change the current angular velocity of the robot's wheels in rad/s
        """
        if target == 'left':
            self.env.set_target_velocity(self._motor_handles[0], velocities)
        if target == 'right':
            self.env.set_target_velocity(self._motor_handles[1], velocities)
        else:
            [self.env.set_target_velocity(self._motor_handles[i], velocities[i]) for i in range(2)]
        
    def find_closest(self):
        """
        Returns [[angle_l, angle_r], [distance_l, distance_r]] of the closest objects in the left and right visual fields
        """
        midpoint = (self._lidar_points) / 2
        sensor_scan_angle = self._lidar_angle / self._lidar_points
        left  = self.lidar_data[:midpoint]
        right = self.lidar_data[midpoint:]
        min_left  = min(left)
        min_right = min(right)
        ind_left  = [i for i, x in enumerate(left)  if x == min_left ][0]
        ind_right = [i for i, x in enumerate(right) if x == min_right][0]
        angle_l = -self._lidar_angle / 2 + ind_left * sensor_scan_angle
        angle_r = 0 + ind_right * sensor_scan_angle
        
        return [[angle_l, angle_r], [min_left, min_right]]

    def read_lidars(self, mode='buffer'):
        """
        Read the vision sensor in VREP
        """
        lidar_data = [self.env.read_lidar(x, mode)[2][1][1::4] for x in self._lidar_handles]
        self.lidar_data = lidar_data[0] + lidar_data[1]
        del self.lidar_data[0]
        del self.lidar_data[0]
        if settings.draw_dist:
            plt.cla()
            plt.plot(self.lidar_data)
            plt.show()
            plt.pause(0.0001)
        return self.lidar_data

    def slam(self, bytearray):
        """
        Get input from sensors and perform SLAM
        """
        # Mapping
        scan = self.read_lidars()
        if self._lidar is None:
            self.init_lidar()
        scan = list(np.array(scan) * settings.scale_factor)
        self._slam_engine.update(scan)
        self._slam_engine.getmap(bytearray) # Draw current map on the bytearray

        # Localization
        x, y, theta = self._slam_engine.getpos()
        self.pos[0] = int(x / float(settings.map_size * settings.agent_scale_factor) * settings.image_size)
        self.pos[1] = int(y / float(settings.map_size * settings.agent_scale_factor) * settings.image_size)
        self.position_history.append(tuple(self.pos)) # Append the position deque with a tuple of (x,y)
        self.theta = theta


class PiCarX(object):
    def __init__(self, env):
        self.env = env
        
        # motors, positions and angles
        self.cam_handle = self.env.get_handle('Vision_sensor')
        self._motor_names = ['Pioneer_p3dx_leftMotor', 'Pioneer_p3dx_rightMotor']
        self._motor_handles = [self.env.get_handle(x) for x in self._motor_names]
        self.angular_velocity = np.zeros(2)
        self.angles = np.zeros(2)
        self.pos = [0, 0]
        self.change_velocity([2, 2])

    def current_speed(self):
        """
        Current angular velocity of the wheel motors in rad/s
        """
        prev_angles = np.copy(self.angles)
        self.angles = np.array([self.env.get_joint_angle(x, 'buffer') for x in self._motor_handles])
        angular_velocity = self.angles - prev_angles
        for i, v in enumerate(angular_velocity):
            # in case radians reset to 0
            if v < -np.pi:
                angular_velocity[i] =  np.pi*2 + angular_velocity[i]
            if v > np.pi:
                angular_velocity[i] = -np.pi*2 + angular_velocity[i]
        self.angular_velocity = angular_velocity
        return self.angular_velocity

    def current_speed_API(self):
        self.angular_velocity = np.array([self.env.get_joint_velocity(x, 'buffer') for x in self._motor_handles])
        return self.angular_velocity

    def change_velocity(self, velocities, target=None):
        """
        Change the current angular velocity of the robot's wheels in rad/s
        """
        if target == 'left':
            self.env.set_target_velocity(self._motor_handles[0], velocities)
        if target == 'right':
            self.env.set_target_velocity(self._motor_handles[1], velocities)
        else:
            [self.env.set_target_velocity(self._motor_handles[i], velocities[i]) for i in range(2)]
    
    def read_image(self, mode='blocking'):
        res, resolution, image = self.env.get_vision_image(self.cam_handle, mode)
        print(res, resolution, image, sep='\n')