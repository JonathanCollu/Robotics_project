from torch import nn
import torch

class PolicyNet(nn.Module):
    """ Neural network acting as policy function for the robot
        agent. The input of the net are two binary maps, where 1s
        indicate that an object/a border is present in those
        locations (relatively to the original image), and 0s are
        assigned to non interesting locations. There are two output
        heads, one with a single sigmoid-activated node indicating
        wether the robot has to move forward or stay still, and the
        other representing the trajectory angle in [0, 180] as a
        classification problem using the softmax function.
    """
    def __init__(self, input_size):
        super(PolicyNet, self).__init__()

        self.hidden_layers = nn.Sequential(
            nn.AvgPool2d(6, 6),
            nn.Flatten(),
            nn.Linear(2400, 256),
            nn.Tanh()
        )

        self.movement_head = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.angle_head = nn.Sequential(
            nn.Linear(256, 180),
            nn.Softmax(dim=1)
        )

    def forward(self, x, device):
        x = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
        x = self.hidden_layers(x)
        movement = self.movement_head(x)
        angle = self.angle_head(x)
        return movement, angle