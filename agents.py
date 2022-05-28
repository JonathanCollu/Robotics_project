import cv2
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
    def __init__(self, policy, optimizer, value_net, optimizer_v, cuboids_num):
        self.env = VrepEnvironment(settings.SCENES + '/environment.ttt')
        self.env.connect()
        self.policy = policy
        self.optimizer = optimizer
        self.value_net = value_net
        self.optimizer_v = optimizer_v
        self.area_min = (-2, -2)
        self.area_max = (2, 2)
        # self.area_min = (-1.25, -1.25)
        # self.area_max = (1.25, 1.25)
        self.pos_decimals = 3
        self.cuboids_num = cuboids_num
        self.cuboids_handles = []
        for i in range(cuboids_num):
            self.cuboids_handles.append(self.env.get_handle(f"Cuboid_{i}"))
        self.set_cuboids_pos()
        self.last_cuboids_mask = None

        # compute attention mask
        attention_center = np.ones((368, 480))
        attention_center[367, int(479*0.5)] = attention_center[367, int(479*0.5)+1] = 0
        self.attention_mask = cv2.distanceTransform((attention_center*255).astype(np.uint8), cv2.DIST_L2, 3)
        self.attention_mask /= self.attention_mask.max()
        self.attention_mask = 1 - self.attention_mask

        # motors, positions and angles
        self.car_handle = self.env.get_handle("Pioneer_p3dx")
        self.cam_handle = self.env.get_handle('Vision_sensor')
        self._motor_names = ['Pioneer_p3dx_leftMotor', 'Pioneer_p3dx_rightMotor']
        self._motor_handles = [self.env.get_handle(x) for x in self._motor_names]
        self.angles = [0, 76, 83, 90, 97, 104, 180]
        # self.forward_vel = (1.2, 1.2)
        self.forward_vel = (6, 6)  # speedrun
        self.stuck_steps = 0

    def set_cuboids_pos(self):
        self.cuboids = []
        for cuboid_handle in self.cuboids_handles:
            pos = self.env.get_object_position(cuboid_handle)
            pos = [round(pos[0], self.pos_decimals), round(pos[1], self.pos_decimals), pos[2]]
            self.cuboids.append(pos)

    def is_pos_allowed(self, i, pos):
        bot_pos = self.env.get_object_position(self.car_handle)
        bot_x = 5.1901e-01 / 2  # bot height
        bot_y = 4.1500e-01 / 2  # bot width
        hs = 1.2500e-01 / 2  # half side
        for j in range(i):
            c2 = self.cuboids[j]
            if c2[0] - hs >= pos[0] - hs and c2[0] - hs <= pos[0] + hs:
                return False, 0
            if c2[0] + hs >= pos[0] - hs and c2[0] + hs <= pos[0] + hs:
                return False, 0
            if c2[1] - hs >= pos[1] - hs and c2[1] - hs <= pos[1] + hs:
                return False, 1
            if c2[1] + hs >= pos[1] - hs and c2[1] + hs <= pos[1] + hs:
                return False, 1
        if pos[0] - bot_x >= bot_pos[0] - bot_x and pos[0] - bot_x <= bot_pos[0] + bot_x:
            return False, 0
        if pos[0] + bot_x >= bot_pos[0] - bot_x and pos[0] + bot_x <= bot_pos[0] + bot_x:
            return False, 0
        if pos[1] - bot_y >= bot_pos[1] - bot_y and pos[1] - bot_y <= bot_pos[1] + bot_y:
            return False, 1
        if pos[1] + bot_y >= bot_pos[1] - bot_y and pos[1] + bot_y <= bot_pos[1] + bot_y:
            return False, 1
        return True, None

    def randomize_positions(self):   
        # randomize orientation of the agent
        self.env.set_object_orientation(self.car_handle, (0, 0, np.random.randint(0, 360)))
        
        # randomize cuboids positions
        for i, cuboid_handle in enumerate(self.cuboids_handles):
            p0 = np.random.uniform(self.area_min[0] + 0.1, self.area_max[0] - 0.1)
            p1 = np.random.uniform(self.area_min[1] + 0.1, self.area_max[1] - 0.1)
            a = 0
            while True:
                a += 1
                if a >= 100: exit()
                pos_allowed = self.is_pos_allowed(i, [p0, p1])
                if pos_allowed[0]: break
                elif pos_allowed[1] == 0:
                    p0 = np.random.uniform(self.area_min[0] + 0.1, self.area_max[0] - 0.1)
                elif pos_allowed[1] == 1:
                    p1 = np.random.uniform(self.area_min[1] + 0.1, self.area_max[1] - 0.1)
            self.cuboids[i] = (p0, p1, self.cuboids[i][2])
            self.env.set_object_position(cuboid_handle, self.cuboids[i])

    def change_velocity(self, velocities, target=None):
        """ Change the current angular velocity of the robot's wheels in rad/s
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
        
        # save screenshot
        # image = Image.fromarray(img)
        # image.save('images/screenshot.png')

        return interpret_image("green", "blue", img)
    
    def start_sim(self, connect):
        if connect:
            self.env.connect()
        self.env.start_simulation()
        # it is required twice more by the sim
        self.env.start_simulation()
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
        # self.randomize_positions()
        self.stuck_steps = 0

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

    def get_reward(self, s_next):
        r = 0
        # compute reward based on the positions of the moved cuboids
        moved_cuboid = False  # has moved at least a cuboid
        cleaned_cuboid = False  # has moved at least a cuboid out of the area
        for i, cuboid_handle in enumerate(self.cuboids_handles):
            # check if the cuboid was already outside the area
            if not self.is_in_area(self.cuboids[i]):
                    continue
            pos = self.env.get_object_position(cuboid_handle)
            pos = [round(pos[0], self.pos_decimals), round(pos[1], self.pos_decimals), pos[2]]
            # check if the cuboid has been moved
            if pos[0] != self.cuboids[i][0] or pos[1] != self.cuboids[i][1]:
                moved_cuboid = True
                # check if the cuboid is outside the area
                if not self.is_in_area(pos):
                    pos = (pos[0], pos[1], -2)
                    self.env.set_object_position(cuboid_handle, pos)
                    r += 10
                    cleaned_cuboid = True
                else:
                    movement_gain = self.min_border_dist(self.cuboids[i]) - self.min_border_dist(pos)
                    # reward between 0 and 1 based on the min distance of the cube to the borders
                    if movement_gain > 0:
                        r += 1 - (self.min_border_dist(pos) / ((self.area_max[0]-self.area_min[0])/2))
                    elif movement_gain < 0:
                        r -= self.min_border_dist(pos) / ((self.area_max[0]-self.area_min[0])/2)
                    # print("cuboid", i, "changed")
                # update stored cuboid position
                self.cuboids[i] = pos
        if not moved_cuboid:
            r -= 0.25

        # compute reward based on the optimality of the cuboids mask in s_next
        # 8000 was obtained as the sum of the product of a near optimal cuboids mask and the
        # attention mask => the fraction means how good the new mask is compared to a good one
        r_cuboids_mask = (s_next[0] * self.attention_mask).sum() / 8000
        if self.last_cuboids_mask is None:
            last_r_cuboids_mask = r_cuboids_mask
        else:
            last_r_cuboids_mask = (self.last_cuboids_mask * self.attention_mask).sum() / 8000
        r += r_cuboids_mask - last_r_cuboids_mask
        self.last_cuboids_mask = s_next[0]
        
        return r, cleaned_cuboid

    def avoid_stuck(self, max_diff, duration):
        # TODO update to new movement method
        print("The robot is stuck")
        self.stuck_steps = 0
        self.change_velocity((-self.forward_vel[0], -self.forward_vel[1]))
        time.sleep(duration)
        angle = 300
        diff = abs(angle - 90) / 90 * max_diff
        self.change_velocity((diff, 0))
        time.sleep(duration)

    def move(self, movement, angle, duration=0.5):
        # set angle and base velocity
        angle = self.angles[angle]
        if not movement:
            # to avoid having the robot stuck, the "stay still" action
            # is replaced with "go forward" (i.e. movement=1 angle=90)
            if angle == 90:
                base = self.forward_vel
            else:
                base = (0, 0)
        else:
            base = self.forward_vel

        # store position of agent before action
        start_pos = self.env.get_object_position(self.car_handle)
        start_pos = [round(start_pos[0], self.pos_decimals), round(start_pos[1], self.pos_decimals)]
        
        # perform action in the environment
        # max_v = ??
        max_v = 6  # speedrun
        v = abs(angle - 90) / 90 * max_v
        if angle > 90:
            self.change_velocity((v, -v))
        elif angle < 90:
            self.change_velocity((-v, v))
        time.sleep(0.235)
        if base != (0, 0):
            self.change_velocity(base)
            time.sleep(0.265)
        self.change_velocity((0, 0))

        # get new state observation
        s_next = self.detect_objects()

        # check for new reward based on the cuboids after the action
        r, cleaned_cuboid = self.get_reward(s_next)

        # check if the agent got stuck, in that case neg. reward and done=True
        end_pos = self.env.get_object_position(self.car_handle)
        end_pos = [round(end_pos[0], self.pos_decimals), round(end_pos[1], self.pos_decimals)]
        if end_pos == start_pos:
            self.stuck_steps += 1
            if self.stuck_steps >= 10:
                r -= 1
                self.stuck_steps = 0
                # self.avoid_stuck(max_v, duration)
                done = True

        # check if agent moved outside of the area
        if not self.is_in_area(self.env.get_object_position(self.car_handle)):
            r -= 1
            done = True
        else:  # check if task is done (all cuboids fell outside the area)
            done = all([self.cuboids[i][2] < 0 for i in range(self.cuboids_num)])
        
        # cut the trace after moving out a cuboid (to learn not to go outside afterwards)
        if cleaned_cuboid and not done:
            done = None

        return s_next, r, done
    
    def act(self, trials):
        self.start_sim(connect=False)
        r_ep = []
        for i in range(trials):
            r_ep.append(0)
            done = False
            self.reset_env()
            s_old = self.detect_objects()
            while not done:
                with torch.no_grad():
                    self.policy.eval()
                    s = self.detect_objects()
                    movement_prob, angles_dist = self.policy.forward(np.vstack((s, s_old)))
                    m = movement_prob.round()
                    a = angles_dist.argmax()
                    s_next, r, done = self.move(m, a)
                    s_old = s
                    s = s_next
                    cub = Image.fromarray(np.array(s[0]*255, np.uint8))
                    cub.save('images/cuboids_mask.png')
                    bor = Image.fromarray(np.array(s[1]*255, np.uint8))
                    bor.save('images/border_mask.png')
                r_ep[i] += r
        return np.mean(r_ep)

    def calibrate(self):
        # testing for movement calibration
        self.start_sim(connect=False)
        base = (6, 6)
        # base = (0, 0)
        angle = 0
        for i in range(1):
            # max_v = ??
            max_v = 6
            v = abs(angle - 90) / 90 * max_v
            if angle > 90:
                self.change_velocity((v, -v))
            elif angle < 90:
                self.change_velocity((-v, v))
            time.sleep(0.235)
            if base != (0, 0):
                self.change_velocity(base)
                time.sleep(0.265)
            self.change_velocity((0, 0))
        time.sleep(2)

    def train(self, epochs, M, T, gamma, ef=None, run_name=None):
        print('Starting training...')
        reinforce = Reinforce(self, epochs, M, T, gamma, ef, run_name)
        rewards = reinforce()
        return rewards
        # self.calibrate()