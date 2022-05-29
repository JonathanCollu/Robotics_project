from src.agents import PiCarX
from src.Model import ConvQNet
import settings
import time
import matplotlib.pyplot as plt
import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-run_name', action='store', type=str, default=None)
    parser.add_argument('-cp_name', action='store', type=str, default=None)
    parser.add_argument('-rb_size', action='store', type=int, default=10)
    parser.add_argument('-batch_size', action='store', type=int, default=3)
    parser.add_argument('-epochs', action='store', type=int, default=10)
    parser.add_argument('-gamma', action='store', type=float, default=0.99)
    parser.add_argument('-target_model', action='store_true')
    parser.add_argument('-tm_wait', action='store', type=int, default=2)
    parser.add_argument('-policy', action='store', type=str, default="egreedy")
    parser.add_argument('-epsilon', action='store', type=float, 
                        nargs="+", default=[0.1, 0.99, 200.])
    parser.add_argument('-temp', action='store', type=float, default=0.1)
    args = parser.parse_args()


    plt.ion()
    model = ConvQNet()
    if args.cp_name is not None:
       state_dict = torch.load(f"exp_results/{args.cp_name}_weights.pt")
       for name, param in state_dict.items():
            if "hidden_layers.1" in name or "hidden_layers.2" in name:
                model.load_state_dict({name: param}, strict=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss = torch.nn.MSELoss()
    batch_size = min(args.rb_size, args.batch_size)

    if len(args.epsilon) > 1:
        epsilon = tuple(args.epsilon)
    else: epsilon = args.epsilon[0]


    agent = PiCarX(model, optimizer, 3)
    
    try:
        agent.train(rb_size=args.rb_size, batch_size=batch_size, n_episodes=args.epochs, 
        loss=loss, gamma=args.gamma, policy=args.policy, epsilon=epsilon, temp=args.temp, target_model=args.target_model, tm_wait=args.tm_wait, 
        run_name=args.run_name)  
    except KeyboardInterrupt:
        print('\n\nInterrupted! Time: {}s'.format(time.time()))

    agent.stop_sim(disconnect=True)