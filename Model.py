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
    def __init__(self):
        super(PolicyNet, self).__init__()

        self.hidden_layers = nn.Sequential(
            nn.AvgPool2d(12, 12),
            nn.Flatten(),
            nn.Linear(2400, 256),
            nn.ReLU()
        )

        self.movement_head = nn.Sequential(
            # nn.Linear(256, 1),
            # nn.Sigmoid()
            nn.Linear(256, 3),
            nn.Softmax(dim=1)
        )

        self.angle_head = nn.Sequential(
            # nn.Linear(256, 181),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 91),
            nn.Softmax(dim=1)
        )

    def forward(self, x, device="cpu"):
        x = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
        x = self.hidden_layers(x)
        movement = self.movement_head(x)
        angle = self.angle_head(x)
        return movement, angle


class TinyPolicyNet(nn.Module):
    """ REDuced policy network for the robot agent. "Reduced" refers to
        the over 5 times lower number of parameters of this network compared
        to the first PolicyNet (~120k vs ~660k weights respectively).
        The input of the net are two binary maps, where 1s
        indicate that an object/a border is present in those
        locations (relatively to the original image), and 0s are
        assigned to non interesting locations.
    """
    def __init__(self):
        super(TinyPolicyNet, self).__init__()

        self.hidden_layers = nn.Sequential(
            nn.AvgPool2d(20, 20),
            nn.Flatten(),
            nn.Linear(864, 128),
            nn.ReLU()
        )

        self.movement_head = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.angle_head = nn.Sequential(
            nn.Linear(128, 7),
            nn.Softmax(dim=1)
        )

    def forward(self, x, device="cpu"):
        x = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
        hidden = self.hidden_layers(x)
        movement = self.movement_head(hidden)
        angle = self.angle_head(hidden)
        return movement, angle


class ConvPolicyNet(nn.Module):
    """ TODO: add description
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
