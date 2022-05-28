from torch import nn
import torch

class ConvPolicyNet(nn.Module):
    """ TODO: FIX DESCRIPTION
        Neural network acting as policy function for the robot
        agent. The input of the net are two binary maps, where 1s
        indicate that an object/a border is present in those
        locations (relatively to the original image), and 0s are
        assigned to non interesting locations. There are two output
        heads, one with a single sigmoid-activated node indicating
        wether the robot has to move forward or stay still, and the
        other representing the trajectory angle in [0, 180] as a
        classification problem using the softmax function.
    """
    def __init__(self, value=False):
        super(ConvPolicyNet, self).__init__()
        self.value = value

        self.hidden_layers = nn.Sequential(
            nn.AvgPool2d(8),
            nn.Conv2d(4, 8, kernel_size=6, stride=2),
            nn.Conv2d(8, 16, kernel_size=5, stride=1),
            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(1536, 128),
            nn.ReLU()
        )

        if not value:
            self.movement_head = nn.Sequential(
                nn.Linear(128, 1),
                nn.Sigmoid()
            )

            self.angle_head = nn.Sequential(
                nn.Linear(128, 7),
                nn.Softmax(dim=1)
            )
        else:
            self.value_head = nn.Sequential(
                nn.Linear(128, 1)
            )

    def forward(self, x, device="cpu"):
        x = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
        hidden = self.hidden_layers(x)
        if not self.value:
            movement = self.movement_head(hidden)
            angle = self.angle_head(hidden)
            return movement, angle
        else:
            return self.value_head(hidden)[0]