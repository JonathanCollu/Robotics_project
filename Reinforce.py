import os
import torch
import numpy as np
from torch.distributions import Categorical
from PIL import Image, ImageOps

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
        best_r_ep = -np.inf
        best_ep = 0
        saving_delay = 10

        # start training
        self.agent.start_sim(connect=False)
        for epoch in range(self.epochs):
            l, r = self.epoch(epoch % 10 == 0)
            losses.append(l)
            rewards.append(r)
            print(f"[{epoch+1}] Epoch mean loss: {round(l, 4)} | Epoch mean reward: {r}")
            if rewards[-1] >= best_r_ep:
                best_r_ep = rewards[-1]
                best_ep = epoch
                print("New max reward in episode:", best_r_ep)
            if self.run_name is not None and epoch % saving_delay == 0:
                self.save_checkpoint(losses, rewards, epoch, saving_delay)

        if self.run_name is not None:
                self.save_checkpoint(losses, rewards, epoch, 0)

        return rewards

    def save_checkpoint(self, losses, rewards, epoch, saving_delay):
        # remove old weights
        if os.path.isfile(f"exp_results/{self.run_name}_{epoch-saving_delay}_weights.pt"): 
            os.remove(f"exp_results/{self.run_name}_{epoch-saving_delay}_weights.pt")
        # save model
        torch.save(self.agent.policy.state_dict(), f"exp_results/{self.run_name}_{epoch}_weights.pt")
        # save losses and rewards
        np.save("exp_results/"+self.run_name, np.array([losses, rewards]))
    
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

    def epoch(self, reset=False):
        if reset:
            self.agent.reset_env()
        loss = torch.tensor([0], dtype=torch.float32) 
        reward = 0
        for m in range(self.M):
            # if m in [1,2]:  # for testing the reset env after done=True
                # self.agent.reset_env()
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
        torch.nn.utils.clip_grad_norm_(self.agent.policy.parameters(), 10)
        # update weigths
        self.agent.optimizer.step()