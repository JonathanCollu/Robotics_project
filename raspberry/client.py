import argparse
from PiCarX import PiCarX

parser = argparse.ArgumentParser()
parser.add_argument('-host_address', action='store', type=str, default=None)
args = parser.parse_args()


def main():
    picar = PiCarX(args.host_address)
    picar.act()
    print("Task completed")


if __name__ == "__main__":
    main()
