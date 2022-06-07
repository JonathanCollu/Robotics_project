import torch
import socket
import argparse
import numpy as np
from PIL import Image
from Model import ConvPolicyNet as Policy

parser = argparse.ArgumentParser()
parser.add_argument('-parameters', action='store', type=str, default=None)
args = parser.parse_args()


def main():
    print("Starting")
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
            packet = b''
            while len(packet) < 5652480:
                packet += connection.recv(4096)
            p1 = packet[:int(len(packet)/2)]
            p2 = packet[int(len(packet)/2):]
            mask1 = np.frombuffer(p1, dtype=np.float64).reshape((2, 368, 480))
            mask2 = np.frombuffer(p2, dtype=np.float64).reshape((2, 368, 480))
            mask1_cuboid = mask1[0]
            mask1_border = mask1[1]
            mask2_cuboid = mask2[0]
            mask2_border = mask2[1]
            if len(mask1) == 0:
                break
            img = Image.fromarray((mask1_cuboid * 255).astype(np.uint8))
            img.save("images/img_cuboid1_" + str(count) + ".png")
            img = Image.fromarray((mask1_border * 255).astype(np.uint8))
            img.save("images/img_border1_" + str(count) + ".png")
            img = Image.fromarray((mask2_cuboid * 255).astype(np.uint8))
            img.save("images/img_cuboid2_" + str(count) + ".png")
            img = Image.fromarray((mask2_border * 255).astype(np.uint8))
            img.save("images/img_border2_" + str(count) + ".png")
            print(mask1.shape)
            m, a = policy.forward(np.vstack((mask1, mask2)))
            #m, a = policy.forward(mask1)
            print('m', m)
            print('a', a)
            m = int(m.round().item())
            a = int(a.argmax().item())
            print('m', m)
            print('a', a)
            msg = str(m) + ";" + str(a)
            print('msg', msg)
            connection.send(msg.encode())
            count += 1
        s.close()
        print("connection succesfully closed")


if __name__ == "__main__":
    main()
