from __future__ import print_function
from src.Model import PolicyNet
from src.agents import PiCarX
from src.Model   import PolicyNet, RedPolicyNet
import settings
import time
import matplotlib.pyplot as plt
import torch


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

if __name__ == "__main__":
    plt.ion()
    model = RedPolicyNet()
    # state_dict = torch.load(f"exp_results/new_relu_2_3_271to460_weights.pt")
    # model.load_state_dict(state_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    agent = PiCarX(model, optimizer, 1)

    try:
        # agent.train(500, 1, 3, 0.99, ef=None, run_name="91a_1c_relu_2_3")
        agent.train(1, 1, 5, 0.99, ef=None, run_name=None)
    except KeyboardInterrupt:
        print('\n\nInterrupted! Time: {}s'.format(time.time()))

    agent.stop_sim(disconnect=True)