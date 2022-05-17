import os
from pickle import NONE
import torch
import numpy as np
from torch.distributions import Categorical

class Reinforce():
    """ Parameters:
            - agent : robot to be trained
            - epochs : number of epochs 
            - M : number of traces per epoch
            - T : trace length
            - gamma : discount factor
            - entropy_factor : entropy factor
            - run_name : name of the run
    """
    def __init__(self, agent, epochs, M, T, gamma, entropy_factor, run_name):
        self.agent = agent
        self.epochs = epochs
        self.M = M
        self.T = T
        self.gamma = gamma
        self.entropy_factor = entropy_factor
        self.run_name = run_name

    def __call__(self):
        rewards = []
        losses = []
        best_r_ep = 0
        best_ep = 0

        # start training
        for epoch in range(self.epochs):
            l, r = self.epoch()
            losses.append(l)
            rewards.append(r)
            print(f"[{epoch+1}] Epoch mean loss: {round(l, 4)} | Epoch mean reward: {r}")
            if rewards[-1] >= best_r_ep:
                best_r_ep = rewards[-1]
                print("New max number of steps in episode:", best_r_ep)
                if self.run_name is not None:
                    # remove old weights
                    if os.path.isfile(f"{self.run_name}_{best_ep}_weights.pt"): 
                        os.remove(f"{self.run_name}_{best_ep}_weights.pt")
                    # save model
                    torch.save(self.agent.policy.state_dict(), f"{self.run_name}_{epoch}_weights.pt")
                    best_ep = epoch
        
        if self.run_name is not None:
            # save losses and rewards
            np.save(self.run_name, np.array([losses, rewards]))
        return rewards
    
    def select_action(self, s):
        # get the probability distribution of the actions
        movement_prob, angles_dist = self.agent.policy.forward(s)
        # sample movement=1 with prob=movement_prob
        movement = torch.bernoulli(movement_prob.detach()).type(torch.int64)
        movement_dist = (movement_prob, 1-movement_prob)
        # sample angle from angles distribution
        angles_dist = Categorical(angles_dist)
        angle = angles_dist.sample()
        return movement, movement_dist, angle, angles_dist

    def sample_trace(self):
        reward = 0
        trace = []
        for _ in range(self.T):
            s = self.agent.detect_objects()
            m, m_dist, a, a_dist = self.select_action(s)
            r, done = self.agent.move(m.item(), a.item())
            trace.append((s, (m, a), r, (m_dist, a_dist)))
            reward += r
            if done:
                self.agent.reset_env()
                break
        trace.append((s, None, None, None))
        self.agent.change_velocity((0, 0))
        return trace, reward

    def epoch(self):
        loss = torch.tensor([0], dtype=torch.float32) 
        reward = 0
        for _ in range(self.M):
            # if _ == 1:  # for testing the reset env after done=True
            #     self.agent.reset_env()
            h0, reward_t = self.sample_trace()
            reward += reward_t
            R = 0
            # len-2 reason: -1 for having 0..len-1 and -1 for skipping last state
            for t in range(len(h0) - 2, -1, -1):
                R = h0[t][2] + self.gamma * R
                loss_m = -torch.log(h0[t][3][0][h0[t][1][0]])[0]
                loss_a = -h0[t][3][1].log_prob(h0[t][1][1])
                loss += R * (loss_m + loss_a)
                if self.entropy_factor is not None:
                    loss += self.entropy_factor * (h0[t][3][0].entropy() + h0[t][3][1].entropy())
        loss /= self.M
        reward /= self.M
        self.train(loss)
        return loss.item(), reward

    def train(self, loss):
        # set model to train
        self.agent.policy.train()
        # compute gradient of loss
        self.agent.optimizer.zero_grad()
        loss.backward()
        # clip gradient
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        # update weigths
        self.agent.optimizer.step()