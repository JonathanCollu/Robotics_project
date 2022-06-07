from src.env import VrepEnvironment
import settings
from src.libs.sim.simConst import sim_boolparam_display_enabled

if __name__ == "__main__":

    environment = VrepEnvironment(settings.SCENES + '/environment.ttt')
    environment.start_vrep(headless=True)
