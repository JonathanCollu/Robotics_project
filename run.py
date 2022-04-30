"""
"""

from __future__ import print_function
from src.env    import VrepEnvironment
from src.agents import Pioneer
from src.disp   import Display
import settings
import time, argparse
import matplotlib.pyplot as plt
from numpy import pi

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

###########
###########

def loop_(agent, started):
    def rotate_and_move(ls, rs, fs, rt, mt):
        '''
            rotates the robot in the direction of the negative speed
            between ls and rs for rt seconds. Then moves forward with
            speed ss for mt seconds
            Parameters:
                ls: left speed
                rs: right speed
                fs: forward speed
                rt: rotation time
                mt: moving (forward) time
        '''
        agent.change_velocity([ls, rs])
        time.sleep(rt)
        agent.change_velocity([fs, fs])
        time.sleep(mt)

    if not started: rotate_and_move(pi, -pi, 5, 1.5, 7)
    left_speed, right_speed = tuple(agent.current_speed_API())
    if (round(left_speed), round(right_speed)) == (0, 0): rotate_and_move(-pi, pi, 3, 1, 0.5)
    elif agent.read_lidars()[62] > 2 : rotate_and_move(pi, -pi, 3, 1.5, 2)

def loop(agent):
    agent.read_image()
    exit()
    
##########
##########

if __name__ == "__main__":
    plt.ion()
    # Initialize and start the environment
    environment = VrepEnvironment(settings.SCENES + '/environment.ttt')  # Open the file containing our scene (robot and its environment)
    environment.connect()        # Connect python to the simulator's remote API
    agent   = PiCarX(environment)
    display = Display(agent, False) 

    print('\nDemonstration of Simultaneous Localization and Mapping using CoppeliaSim robot simulation software. \nPress "CTRL+C" to exit.\n')
    start = time.time()
    step  = 0
    done  = False
    environment.start_simulation()
    time.sleep(1)

    try:    
        while step < settings.simulation_steps and not done:
            display.update()                     # Update the SLAM display
            
            loop(agent, step != 0)                           # Control loop
            step += 1
    except KeyboardInterrupt:
        print('\n\nInterrupted! Time: {}s'.format(time.time()-start))
        
    display.close()
    environment.stop_simulation()
    environment.disconnect()
