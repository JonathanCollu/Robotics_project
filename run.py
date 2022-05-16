from __future__ import print_function
from src.Model import PolicyNet
from src.agents import PiCarX
from src.Model   import PolicyNet
import settings
import time
import matplotlib.pyplot as plt


""" Motors:

          1. agent.change_velocity([ speed_left: float, speed_right: float ]) 
               Set the target angular velocities of left
               and right motors with a LIST of values:
               e.g. [1., 1.] in radians/s.
               
               Values in range [-5:5] (above these 
               values the control accuracy decreases)
                    
          2. agent.current_speed_API() 
          ----
               Returns a LIST of current angular velocities
               of the motors
               [speed_left: float, speed_right: float] in radians/s.

    Lidar:
          3. agent.read_lidars()   
          ----
               Returns a list of floating point numbers that you can 
               indicate the distance towards the closest object at a particular angle.
               
               Basic configuration of the lidar:
               Angle: [-135:135] Starting with the 
               leftmost lidar point -> clockwise

    Agent:
          You can access these attributes to get information about the agent's positions

          4. agent.pos  

          ----
               Current x,y position of the agent (derived from 
               SLAM data) Note: unreliable as SLAM is not solved here.

          5. agent.position_history

               A deque containing N last positions of the agent 
               (200 by default, can be changed in settings.py) Note: unreliable as SLAM is not solved here.
"""

def loop(agent):
    agent.train(10, 2, 500, 0.99)

if __name__ == "__main__":
    plt.ion()
    model = PolicyNet()
    agent   = PiCarX(model)

    agent.reset_env()
    time.sleep(1)

    try:    
        loop(agent)  # Control loop
    except KeyboardInterrupt:
        print('\n\nInterrupted! Time: {}s'.format(time.time()))

    agent.env.stop_simulation()
    try: agent.env.disconnect()
    except: pass