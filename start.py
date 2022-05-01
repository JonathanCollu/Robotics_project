from src.env import VrepEnvironment
import settings

if __name__ == "__main__":

    environment = VrepEnvironment(settings.SCENES + '/environment.ttt')
    environment.start_vrep()