import torch
import socket
import argparse
import numpy as np
from PIL import Image
from Model import PolicyNet as Policy

parser = argparse.ArgumentParser()
parser.add_argument('-parameters', action='store', type=str, default=None)
args = parser.parse_args()

def main():
    policy = Policy()
    state_dict = torch.load(f"exp_results/" + args.parameters)
    policy.load_state_dict(state_dict)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 4242))
    s.listen()
    connection, _ = s.accept()
    count = 0
    with connection:
        while 1: 
            mask = np.frombuffer(connection.recv(1024), dtype=np.uint8)
            if len(mask) == 1: break
            img = Image.fromarray(mask)
            img.save("images/img_" + str(count) + ".png")
            m, a = policy.forward(mask)
            m = m.round()
            a = a.argmax()
            msg = str(m) + ";" + str(a)
            s.send(msg.encode())
            count += 1
        s.close()
        print("connection succesfully closed")



if __name__ == "__main__":
    main()
