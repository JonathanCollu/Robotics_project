from src.agents import PiCarX
from src.Model import ConvPolicyNet
import settings
import time
import matplotlib.pyplot as plt
import torch

if __name__ == "__main__":
    plt.ion()
    policy = ConvPolicyNet()
    state_dict = torch.load(f"exp_results/conv4ch_base_10c_1_7_1981_weights.pt")
    for name, param in state_dict.items():
        if "hidden_layers.1" in name or "hidden_layers.2" in name:
            policy.load_state_dict({name: param}, strict=False)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    value_net = ConvPolicyNet(value=True)
    state_dict = torch.load(f"exp_results/conv4ch_base_10c_1_7_1981_weights.pt")
    for name, param in state_dict.items():
        if "hidden_layers.1" in name or "hidden_layers.2" in name:
            value_net.load_state_dict({name: param}, strict=False)
    optimizer_v = torch.optim.Adam(value_net.parameters(), lr=1e-3)

    agent = PiCarX(policy, optimizer, value_net, optimizer_v, 10)

    try:
        agent.train(1000, 1, 7, 0.99, ef=None, run_name="conv4ch_base_10c_1_7")
    except KeyboardInterrupt:
        print('\n\nInterrupted! Time: {}s'.format(time.time()))

    agent.stop_sim(disconnect=True)