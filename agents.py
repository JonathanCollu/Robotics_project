from tabnanny import check
import time

import torch
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
    def __init__(self, policy, optimizer, cuboids_num):
        self.env = VrepEnvironment(settings.SCENES + '/environment.ttt')
        # self.start_sim(connect=True)
        self.env.connect()
        self.policy = policy
        self.optimizer = optimizer
        self.area_min = (-2, -2)
        self.area_max = (2, 2)
        self.cuboids_num = cuboids_num
        self.cuboids_handles = []
        for i in range(cuboids_num):
            self.cuboids_handles.append(self.env.get_handle(f"Cuboid_{i}"))
        self.set_cuboids_pos()

        # motors, positions and angles
        self.car_handle = self.env.get_handle("Pioneer_p3dx")
        self.cam_handle = self.env.get_handle('Vision_sensor')
        self._motor_names = ['Pioneer_p3dx_leftMotor', 'Pioneer_p3dx_rightMotor']
        self._motor_handles = [self.env.get_handle(x) for x in self._motor_names]
        self.angular_velocity = np.zeros(2)
        self.angles = np.zeros(2)
        self.forward_vel = (1.2, 1.2)
        self.stuck_steps = 0

    def set_cuboids_pos(self):
        self.cuboids = []
        for cuboid_handle in self.cuboids_handles:
            pos = self.env.get_object_position(cuboid_handle)
            pos = [round(pos[0], 2), round(pos[1], 2)]
            self.cuboids.append(pos)

    def check_pos(self, i, c):
        bot_pos = self.env.get_object_position(self.car_handle)
        bot_x = 5.1901e-01 / 2
        bot_y = 4.1500e-01 / 2
        hs = 1.2500e-01 / 2
        for j in range(i):
            if j == i: break
            c2 = self.cuboids[j]
            if c2[0] - hs >= c[0] - hs and c2[0] - hs <= c[0] + hs:
                return True
            if c2[0] + hs >= c[0] - hs and c2[0] + hs <= c[0] + hs:
                return True
            if c2[1] - hs >= c[1] - hs and c2[1] - hs <= c[1] + hs:
                return True
            if c2[1] + hs >= c[1] - hs and c2[1] + hs <= c[1] + hs:
                return True
        if c[0] - bot_x >= bot_pos[0] - bot_x and c[0] - bot_x <= bot_pos[0] + bot_x:
            return True
        if c[0] + bot_x >= bot_pos[0] - bot_x and c[0] + bot_x <= bot_pos[0] + bot_x:
            return True
        if c[1] - bot_y >= bot_pos[1] - bot_y and c[1] - bot_y <= bot_pos[1] + bot_y:
            return True
        if c[1] + bot_y >= bot_pos[1] - bot_y and c[1] + bot_y <= bot_pos[1] + bot_y:
            return True
        return False


        

    def randomize_positions(self):
        for i, cuboid_handle in enumerate(self.cuboids_handles):
            p0 = np.random.uniform(self.area_min[0] + 0.5, self.area_max[0] - 0.5)
            p1 = np.random.uniform(self.area_min[1] + 0.5, self.area_max[1] - 0.5)
            while True:
                overlapped = self.check_pos(i,[p0,p1])
                if not overlapped: break
                p0 = np.random.uniform(self.area_min[0] + 0.5, self.area_max[0] - 0.5)
                p1 = np.random.uniform(self.area_min[1] + 0.5, self.area_max[1] - 0.5)
            self.env.set_object_position(cuboid_handle, [p0,p1])
            self.set_cuboids_pos()
                



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
            # print(img, res, sep="\n")
            exit("Error during detection")
        img = np.flip(img, axis=0)
        
        image = Image.fromarray(img)
        image.save('images/screenshot.png')

        return interpret_image("green", "blue", img)
    
    def save_image(self, image, resolution, options, filename, quality=-1):
        self.env.save_image(image, resolution, options, filename, quality)
    
    def start_sim(self, connect):
        if connect:
            self.env.connect()
        self.env.start_simulation()
        # we don't know why, but it is required twice more
        self.env.start_simulation()
        self.env.start_simulation()

    def stop_sim(self, disconnect):
        self.env.stop_simulation()
        if disconnect:
            self.env.disconnect()

    def reset_env(self, connect=False):
        try: self.stop_sim(connect)
        except: pass
        self.start_sim(connect)
        self.set_cuboids_pos()
        self.randomize_positions()

    def min_border_dist(self, point):
        """ minimum distance between a point and the 4 borders
        """
        return min(self.area_max[0]-point[0],  # right border
                   self.area_max[1]-point[1],  # top border
                   point[0]-self.area_min[0],  # left border
                   point[1]-self.area_min[1])  # bottom border

    def is_in_area(self, pos):
        if pos[0] < self.area_min[0] or pos[0] > self.area_max[0]:
            return False
        if pos[1] < self.area_min[1] or pos[1] > self.area_max[1]:
            return False
        return True

    def get_reward(self, cuboids_mask):
        r = 0
        # add cuboids reward
        success = False
        for i, cuboid_handle in enumerate(self.cuboids_handles):
            # check if the cuboid was already outside the area
            if not self.is_in_area(self.cuboids[i]):
                    continue
            pos = self.env.get_object_position(cuboid_handle)
            pos = [round(pos[0], 2), round(pos[1], 2)]
            # check if the cuboid has been moved
            if pos[0] != self.cuboids[i][0] or pos[1] != self.cuboids[i][1]:
                success = True
                # check if the cuboid is outside the area
                if not self.is_in_area(pos):
                    out_of_area_pos = (self.cuboids[i][0], self.cuboids[i][1], -2)
                    self.env.set_object_position(cuboid_handle, out_of_area_pos)
                    r += 10
                else:
                    movement_gain = self.min_border_dist(self.cuboids[i]) - self.min_border_dist(pos)
                    # reward between 0 and 1 based on the min distance of the cube to the borders
                    if movement_gain > 0:
                        r += 1 - (self.min_border_dist(pos) / ((self.area_max[0]-self.area_min[0])/2))
                    elif movement_gain < 0:
                        r -= self.min_border_dist(pos) / ((self.area_max[0]-self.area_min[0])/2)
                    else:
                        r += 0
                    # print("cuboid", i, "changed")
                # update stored cuboid position
                self.cuboids[i] = pos
        
        r += np.sum(cuboids_mask) / cuboids_mask.size
        if not success:
            # r -= 0.1 - np.sum(cuboids_mask) / cuboids_mask.size
            if np.sum(cuboids_mask) / cuboids_mask.size == 0:
                r -= 0.1
        return r

    def move(self, movement, angle, s, duration=1):
        # move the robot in env and return the collected reward
        if not movement:
            # to avoid having the robot stuck, the "stay still" action
            # is replaced with "go forward" (i.e. movement=1 angle=90)
            if angle == 90:
                base = self.forward_vel
            else:
                base = (0, 0)
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
        # check for position outside the area during movement
        # if outside, go back to the last position
        start_pos = self.env.get_object_position(self.car_handle)
        start_pos = [round(start_pos[0], 1), round(start_pos[1], 1)]
        start_time = time.time()
        outside = False
        while True:
            if not self.is_in_area(self.env.get_object_position(self.car_handle)):
                # print("OUTSIDE")
                outside = True
                self.change_velocity((-v[0], -v[1]))
                # time.sleep((time.time()-start_time))
                time.sleep(1)
                break
            if time.time() - start_time >= duration:
                break
            time.sleep(0.05)
        # check if the robot is stuck
        end_pos = self.env.get_object_position(self.car_handle)
        end_pos = [round(end_pos[0], 1), round(end_pos[1], 1)]
        if end_pos == start_pos:
            if not outside:
                self.stuck_steps += 1
            if self.stuck_steps >= 10:
                # print("The robot is stuck")
                self.stuck_steps = 0
                self.change_velocity((-self.forward_vel[0], -self.forward_vel[1]))
                time.sleep(1)
                angle = 300
                diff = abs(angle - 90) / 90 * max_diff
                self.change_velocity((diff, 0))
                time.sleep(1)
        # reset velocity to (0, 0)
        self.change_velocity((0, 0))
        # check for new rewards
        r = self.get_reward(s[0])
        # penalize if no good action + has moved outside
        if r <= 0 and outside:
            r = -1
        # check if robot fell outside the area
        if self.env.get_object_position(self.car_handle)[2] < 0:
            done = True
        else:  # check if task is done
            done = not any(self.cuboids)
        return r, done
    
    def act(self, trials):
        self.start_sim(connect=False)
        r_ep = []
        for i in range(trials):
            r_ep.append(0)
            done = False
            self.reset_env()
            test=0
            while not done:
                with torch.no_grad():
                    self.policy.eval()
                    s = self.detect_objects()
                    cub = Image.fromarray(np.array(s[0]*255, np.uint8))
                    cub.save('images/cuboids_mask.png')
                    bor = Image.fromarray(np.array(s[1]*255, np.uint8))
                    bor.save('images/border_mask.png')
                    if test==6:
                        self.stop_sim(disconnect=True)
                        break
                        #exit()
                    test+=1
                    movement_prob, angles_dist = self.policy.forward(s)
                    m = movement_prob.round()
                    a = angles_dist.argmax()
                    r, done = self.move(1, 90, s)
                r_ep[i] += r
                print(r)
        return np.mean(r_ep)

    def calibrate(self):
        # testing for movement calibration
        self.start_sim(connect=False)
        # base = (1.2, 1.2)
        base = (0, 0)
        angle = 90
        for i in range(4):
            max_diff = 1.7
            diff = abs(angle - 90) / 90 * max_diff
            if angle > 90:
                diff = (diff, 0)
            elif angle < 90:
                diff = (0, diff)
            else:
                diff = (0, 0)
            v = [base[0] + diff[0], base[1] + diff[1]]
            self.change_velocity(v)
            print(self.get_reward())
            time.sleep(1)
        self.change_velocity((0, 0))
        time.sleep(4)

    def train(self, epochs, M, T, gamma, ef=None, run_name=None):
        print('Starting training...')
        reinforce = Reinforce(self, epochs, M, T, gamma, ef, run_name)
        rewards = reinforce()
        return rewards
        # self.calibrate()