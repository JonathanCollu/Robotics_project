import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from src.color_detection import interpret_image
from agents import PiCarX

import sys, os
import settings
import cv2

from utils import reset_mcu
from picarx import Picarx
from picamera import PiCamera

""" TODO: Probably paths not right """
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.append(r'/home/pi/picar-x/lib') #?
reset_mcu()

if settings.draw_dist:
    plt.ion()
    plt.figure()

class RealPiCarX(PiCarX):
    def __init__(self, policy, weights):
        self.px = Picarx()
        state_dict = torch.load(f"exp_results/" + weights)
        self.policy = policy
        self.policy.load_state_dict(state_dict)
        self.angular_velocity = 0 # Pay attention: with Picarx, this is the motor power
        self.angle = 0 # Pay attention: not in radians
        self.forward_vel = 1.2
        self.stuck_steps = 0
        """ TODO: load policy. """
        self.init_camera()
        
    def init_camera(self):
        self.camera = PiCamera()
        self.camera.resolution = (360,480)
        self.camera.framerate = 24

    def change_velocity(self, velocity):
        """
        Change the current angular velocity of the robot's wheels
        """
        self.angular_velocity = velocity
        self.px.forward(velocity)
        
    def change_angle(self, angle):
        self.angle = angle
        self.px.set_dir_servo_angle(angle)
    
    def read_image(self):
    	# if this one doesn't work properly, try the one below
        image = np.empty((self.camera.resolution[0] * self.camera.resolution[1] * 3,), dtype=np.uint8)
        self.camera.capture(image, 'rgb')
        image = image.reshape((self.camera.resolution[0], self.camera.resolution[1], 3))
        return image, self.camera.resolution
    
    '''
    def read_image(self):
        with PiCamera() as camera:
            camera.resolution = (480, 360)
            camera.framerate = 24
            img = np.empty((480, 360, 3), dtype=np.uint8)
            camera.capture(img, 'rgb')
            return img.reshape((360, 480, 3))
    '''
    
    def save_image(self, image, filename):
        image = image.flip(image, axis=0)
        image = Image.fromarray(image)
        image.save(filename) 

    def detect_objects(self):
        img = self.read_image()
        img = np.flip(img, axis=0)
        return interpret_image("green", "blue", img)
        
    def move(self, movement, angle, rt):
        # angle += 45  # test reduced actions
        # move the robot in env and return the collected reward
        if rt is not None:
            if rt == 1:
                angle = 180 - angle
        if not movement:
            # to avoid having the robot stuck, the "stay still" action
            # is replaced with "go forward" (i.e. movement=1 angle=90)
            base = self.forward_vel if angle == 90 else 0
        else:
            base = self.forward_vel
        max_diff = 1.89
        diff = abs(angle - 90) / 90 * max_diff
        if angle > 90:
            diff = (diff, 0)
        elif angle < 90:
            diff = (0, diff)
        else:
            diff = (0, 0)
        v = (base[0] + diff[0], base[1] + diff[1])
        self.change_velocity(v)
        
    
    def act(self):
        """ TODO: Clarify how policy works. """
        while True:
            with torch.no_grad():
                self.policy.eval()
                s = self.detect_objects()
                cub = Image.fromarray(np.array(s[0]*255, np.uint8))
                cub.save('images/cuboids_mask.png')
                bor = Image.fromarray(np.array(s[1]*255, np.uint8))
                bor.save('images/border_mask.png')
                
                # Click ESC key to stop
                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    break
                
                movement_prob, right_turn_prob, angles_dist = self.policy.forward(s)
                m = movement_prob.round()
                rt = right_turn_prob.round() 
                a = angles_dist.argmax()
                self.change_velocity(m)
                self.change_angle(a)
