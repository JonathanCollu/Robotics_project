from __future__ import print_function
from src.Model import ConvPolicyNet, ConvQNet
from src.agentR import PiCarX as PiCarXR
from src.agents import PiCarX
import settings
import time
import matplotlib.pyplot as plt
import torch

if __name__ == "__main__":
    plt.ion()
    model = ConvPolicyNet().double()
    state_dict = torch.load(f"exp_results/policynet_final_weights.pt")
    model.load_state_dict(state_dict)
    agent = PiCarXR(model, None, None, None, 20)
    # agent = PiCarX(model, None, 20)

    try:
        agent.act(1)
    except KeyboardInterrupt:
        print('\n\nInterrupted! Time: {}s'.format(time.time()))

    agent.stop_sim(disconnect=True)