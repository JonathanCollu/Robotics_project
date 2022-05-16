from color_detecton import interpret_image
from model import PolicyNet
import numpy as np
import torch
import cv2

ball_mask = cv2.imread("ball_mask.png", cv2.IMREAD_GRAYSCALE)
border_mask = cv2.imread("border_mask.png", cv2.IMREAD_GRAYSCALE)
mask = np.stack([ball_mask, border_mask])
print(mask.shape)

policy = PolicyNet()
movement, angle = policy.forward(mask)
print("Move forward:", int(torch.round(movement.detach()).item()))
print("Angle:", torch.argmax(angle.detach()).item())