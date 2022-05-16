from color_detecton import interpret_image
from model import PolicyNet
import numpy as np
import torch
import cv2

src_img = cv2.imread('img-1.png')
mask = interpret_image("green", "red", src_img)
print(mask.shape)

policy = PolicyNet()
movement, angle = policy.forward(mask)
print("Move forward:", int(torch.round(movement.detach()).item()))
print("Angle:", torch.argmax(angle.detach()).item())