from __future__ import print_function
from src.Model import ConvPolicyNet
from src.agents import PiCarX
import settings
import time
import matplotlib.pyplot as plt
import torch

if __name__ == "__main__":
    plt.ion()
    model = ConvPolicyNet()
    state_dict = torch.load(f"exp_results/conv4ch_base_10c_1_7_1981_weights.pt")
    model.load_state_dict(state_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    agent = PiCarX(model, optimizer, None, None, 10)

    try:
        agent.act(1)
    except KeyboardInterrupt:
        print('\n\nInterrupted! Time: {}s'.format(time.time()))

    agent.stop_sim(disconnect=True)