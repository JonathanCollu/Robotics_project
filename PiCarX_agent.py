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
    def __init__(self, policy):
        self.px = Picarx()
        self.policy = policy
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

    def current_speed(self):
        """
        Current angular velocity of the wheel motors
        """
        return self.angular_velocity

    def current_speed_API(self):
        return self.angular_velocity

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
        image = np.empty((self.camera.resolution[0] * self.camera.resolution[1] * 3,), dtype=np.uint8)
        self.camera.capture(image, 'rgb')
        image = image.reshape((self.camera.resolution[0], self.camera.resolution[1], 3))
        return image, self.camera.resolution

    def detect_objects(self):
        img, res = self.read_image()
        img = np.flip(img, axis=0)
        
        image = Image.fromarray(img)
        image.save('images/screenshot.png')

        return interpret_image("green", "blue", img)
    
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
                
                movement_prob, angles_dist = self.policy.forward(s)
                m = movement_prob.round() #??????????????????????????????????????????????????
                a = angles_dist.argmax()
                self.change_velocity(m)
                self.change_angle(a)