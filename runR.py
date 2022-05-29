from src.agents import PiCarX
from src.Model import ConvPolicyNet
import settings
import time
import matplotlib.pyplot as plt
import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-run_name', action='store', type=str, default=None)
    parser.add_argument('-cp_name', action='store', type=str, default=None)
    parser.add_argument('-epochs', action='store', type=int, default=1000)
    parser.add_argument('-M', action='store', type=int, default=1)
    parser.add_argument('-T', action='store', type=int, default=7)
    parser.add_argument('-gamma', action='store', type=float, default=0.99)
    parser.add_argument('-ef', action='store', type=float, default=None)
    args = parser.parse_args()

    plt.ion()
    policy = ConvPolicyNet()
    if args.cp_name is not None:
        state_dict = torch.load(f"exp_results/{args.cp_name}_weights.pt")
        for name, param in state_dict.items():
            if "hidden_layers.1" in name or "hidden_layers.2" in name:
                policy.load_state_dict({name: param}, strict=False)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    value_net = ConvPolicyNet(value=True)
    if args.cp_name is not None:
        state_dict = torch.load(f"exp_results/{args.cp_name}_weights.pt")
        for name, param in state_dict.items():
            if "hidden_layers.1" in name or "hidden_layers.2" in name:
                value_net.load_state_dict({name: param}, strict=False)
    optimizer_v = torch.optim.Adam(value_net.parameters(), lr=1e-3)

    agent = PiCarX(policy, optimizer, value_net, optimizer_v, 10)
    
    try:
        agent.train(args.epochs, args.M, args.T, args.gamma, ef=args.ef, run_name=args.run_name)
    except KeyboardInterrupt:
        print('\n\nInterrupted! Time: {}s'.format(time.time()))

    agent.stop_sim(disconnect=True)