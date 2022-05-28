import time
import cv2
import socket
import pickle
import numpy as np
from PIL import Image
import picar_4wd as fc
#from picarx import Picarx
from picamera import PiCamera
from io import BytesIO
from color_detection import interpret_image


class PiCarX():
    def __init__(self, host):
        #self.px = Picarx()
        self.angular_velocity = 0  # Pay attention: with Picarx, this is the motor power
        self.angle = 0  # Pay attention: not in radians
        self.forward_vel = 10
        self.camera = PiCamera()
        self.camera.resolution = (480, 368)
        self.camera.framerate = 24
        self.camera.start_preview()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((host, 4242))

    def stop_if_done(self):
        # Click ESC key to stop
        k = cv2.waitKey(1) & 0xFF
        return k == 27

    def change_velocity(self, velocity):
        """
        Change the current angular velocity of the robot's wheels
        """
        self.angular_velocity = velocity
        fc.forward(velocity)

    def change_angle(self, angle):
        self.angle = angle
        if angle < 90:
            fc.turn_left(self.forward_vel)
        if angle > 90:
            fc.turn_right(self.forward_vel)

    def read_image(self):
        img = np.empty((480 * 368 * 3,), dtype=np.uint8)
        self.camera.capture(img, 'rgb')
        return img.reshape((368, 480, 3))

    def detect_objects(self):
        img = self.read_image()
        return interpret_image("green", "purple", img)

    def request_action(self, state):
        msg = state.tobytes()
        print(state.dtype, len(msg))
        self.socket.send(msg)
        return self.socket.recv(1024).decode().split(";")

    def move(self, movement, angle):
        """
        # angle += 45  # test reduced actions
        # move the robot in env and return the collected reward
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

        v = base + diff
        print('velocity', v)
        print('angle', angle)
        self.change_velocity(v)
        self.change_angle(angle)
        time.sleep(10)
        """
        if angle == 90:
            fc.forward(self.forward_vel)
        elif angle < 90:
            fc.turn_left(self.forward_vel)
        elif angle > 90:
            fc.turn_right(self.forward_vel)
        time.sleep(10)

    def act(self):
        s_old = self.detect_objects()
        while True:
            if self.stop_if_done():
                break
            s = self.detect_objects()
            out = self.request_action(np.vstack((s, s_old)))
            print(out)
            m = int(out[0])
            a = int(out[1])
            self.move(m, a)
            s_old = s
            self.change_velocity(0)
        self.socket.send(np.zeros(0).tobytes())
        self.socket.close()
        print("connection succesfully closed")
