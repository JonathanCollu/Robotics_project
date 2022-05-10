from __future__ import print_function
from src.env    import VrepEnvironment
from src.agents import PiCarX
from src.disp   import Display
import settings
import time, argparse
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

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

def loop(agent, i):
    img, res = agent.read_image()
    img = np.array(img, dtype=np.uint8).reshape([res[1], res[0], 3])
    img = np.flip(img, axis=0)
    img = Image.fromarray(img)
    img.save('images/img' +str(i -1) + '.png')
    agent.change_velocity([2, 2])

if __name__ == "__main__":
    plt.ion()
    # Initialize and start the environment
    print(settings.SCENES)
    environment = VrepEnvironment(settings.SCENES + '/environment.ttt')  # Open the file containing our scene (robot and its environment)
    environment.connect()        # Connect python to the simulator's remote API
    agent   = PiCarX(environment)

    print('\nDemonstration of Simultaneous Localization and Mapping using CoppeliaSim robot simulation software. \nPress "CTRL+C" to exit.\n')
    start = time.time()
    step  = 0
    done  = False
    environment.start_simulation()
    time.sleep(1)

    try:    
        while step < settings.simulation_steps and not done:
            loop(agent, step)  # Control loop
            step += 1
    except KeyboardInterrupt:
        print('\n\nInterrupted! Time: {}s'.format(time.time()-start))

    environment.stop_simulation()
    environment.disconnect()
