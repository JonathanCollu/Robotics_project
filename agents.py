import time

from matplotlib import image
import torch
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from src.color_detection import interpret_image
from src.Reinforce import Reinforce
from src.env import VrepEnvironment
from src.libs.sim.simConst import sim_boolparam_display_enabled

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import settings

if settings.draw_dist:
    plt.ion()
    plt.figure()

class PiCarX(object):
    def __init__(self, policy, optimizer):
        self.env = VrepEnvironment(settings.SCENES + '/environment.ttt')
        self.start_sim(connect=True)
        self.policy = policy
        self.optimizer = optimizer
        
        # motors, positions and angles
        self.cam_handle = self.env.get_handle('Vision_sensor')
        self._motor_names = ['Pioneer_p3dx_leftMotor', 'Pioneer_p3dx_rightMotor']
        self._motor_handles = [self.env.get_handle(x) for x in self._motor_names]
        self.angular_velocity = np.zeros(2)
        self.angles = np.zeros(2)
        self.pos = [0, 0]
        
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
        _, resolution, image = self.env.get_vision_image(self.cam_handle, mode)
        return image, resolution

    def detect_objects(self):
        img, res = self.read_image()

        try: img = np.array(img, dtype=np.uint8).reshape([res[1], res[0], 3])
        except: 
            print(img, res, sep="\n")
            exit()
        img = np.flip(img, axis=0)
        
        # image = Image.fromarray(img)
        # image.save('images/screenshot.png')

        return interpret_image("green", "red", img)
    
    def save_image(self, image, resolution, options, filename, quality=-1):
        self.env.save_image(image, resolution, options, filename, quality)
    
    def start_sim(self, connect):
        if connect:
            self.env.connect()
        self.env.start_simulation()
        # we don't know why, but it is required twice
        self.env.start_simulation()

    def stop_sim(self, disconnect):
        self.env.stop_simulation()
        if disconnect:
            self.env.disconnect()

    def reset_env(self, connect=False):
        try: self.stop_sim(connect)
        except: pass
        self.start_sim(connect)
    
    def move(self, movement, angle):
        # move the robot in env and return the collected reward
        if not movement:
            base = (0, 0)
        else:
            base = (1.2, 1.2)
        max_diff = 1.7
        diff = abs(angle - 90) / 90 * max_diff
        if angle > 90:
            diff = (diff, 0)
        elif angle < 90:
            diff = (0, diff)
        else:
            diff = (0, 0)
        v = (base[0] + diff[0], base[1] + diff[1])
        self.change_velocity(v)
        time.sleep(1)
        self.change_velocity(base)
        # TODO: implement reward system
        return 0, False
    
    def act(self, trials):
        r_ep = [0]*trials
        for i in range(trials):
            done = False
            self.reset_env()
            while not done:
                with torch.no_grad():
                    self.policy.eval()
                    s = self.detect_objects()
                    pred = self.policy.forward(s)
                    r, done = self.move(torch.argmax(pred.detach()).item())
                r_ep[i] += r
        return np.mean(r_ep)

    def train(self, epochs, M, T, gamma, ef=None, run_name=None):
        print('Starting training...')
        reinforce = Reinforce(self, epochs, M, T, gamma, ef, run_name)
        rewards = reinforce()
        return rewards

        # testing for movement calibration
        # base = (1.2, 1.2)
        # base = (0, 0)
        # angle = 0
        # for i in range(1):
        #     print(i)
        #     max_diff = 1.7
        #     diff = abs(angle - 90) / 90 * max_diff
        #     print(diff)
        #     if angle > 90:
        #         diff = (diff, 0)
        #     elif angle < 90:
        #         diff = (0, diff)
        #     else:
        #         diff = (0, 0)
        #     v = [base[0] + diff[0], base[1] + diff[1]]
        #     self.change_velocity(v)
        #     time.sleep(1)