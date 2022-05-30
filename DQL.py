import os
import random
import cv2
import numpy as np
import torch
from copy import deepcopy
from collections import deque
from PIL import Image
from Utilities import *
from .Model import *


class DQL:
    """ Parameters:
            - use_rb : enables DQN with experience replay buffer
            - rb_size : experience replay buffer size
            - n_episodes : size of the episode to consider
            - n_timesteps : numer of timesteps per episode to consider
            - minibatch_size : size of minibatch to consider
            - epsilon : probability to choose a random action
            - temp : temperature parameter for softmax selection
            - model : Deep Learning agent (model in pytorch)
            - input_is_img : defines if the input is an image
            - env : environment to train our model 
    """
    def __init__(
            self,
            agent,
            loss, 
            use_rb,
            batch_size,
            rb_size,
            n_episodes,
            gamma,
            policy,
            epsilon,
            temp,
            target_model,
            tm_wait,
            run_name
        ):
        self.agent = agent
        self.batch_size = batch_size
        self.use_rb = use_rb
        self.rb_size = rb_size
        self.n_episodes = n_episodes
        self.gamma = gamma
        self.policy = policy
        # tensor of total count for every possible action
        self.actions_count = torch.tensor([1]*len(self.agent.angles), dtype=torch.float32)
        # tensor of total reward for every possible action
        self.epsilon = epsilon
        self.temp = temp
        self.model = self.agent.policy.double()
        # create an identical separated model updated as self.model each episode
        self.target_model = deepcopy(self.model) if target_model else self.model
        self.tm_wait = tm_wait
        self.loss = loss
        self.run_name = run_name

    def __call__(self):
        # create replay buffer
        self.rb = deque([], maxlen=self.rb_size)

        # iterate over episodes
        self.training_started = False
        self.ts_tot = 0
        rewards = []
        losses = []
        best_ep = -np.inf
        saving_delay = 1
        for epoch in range(self.n_episodes):
            l, r = self.episode(epoch, reset=False)
            losses.append(l)
            rewards.append(r)
            print(f"[{epoch+1}] Epoch mean loss: {round(l, 4)} | Epoch mean reward: {r}")
            if rewards[-1] >= best_ep:
                best_ep = rewards[-1]
                if self.run_name is not None:
                    self.save_checkpoint(losses, rewards, epoch, epoch-best_ep)
                best_ep = epoch
                print("New max reward in episode:", best_ep)
            if self.run_name is not None and epoch % saving_delay == 0:
                self.save_checkpoint(losses, rewards, epoch, saving_delay)
    

    def save_checkpoint(self, losses, rewards, epoch, saving_delay):
        # remove old weights
        if os.path.isfile(f"exp_results/{self.run_name}_{epoch-saving_delay}_weights.pt"): 
            os.remove(f"exp_results/{self.run_name}_{epoch-saving_delay}_weights.pt")
        # save model
        torch.save(self.agent.policy.state_dict(), f"exp_results/{self.run_name}_{epoch}_weights.pt")
        # save losses and rewards
        np.save("exp_results/"+self.run_name, np.array([losses, rewards]))

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def load_pretrained(self, path):
        state_dict = torch.load(path)
        for name, param in state_dict.items():
            if "output_head" in name:
                continue
            self.model.load_state_dict({name: param}, strict=False)

    def training_step(self):
        # draw batch of experiences from replay buffer
        sampled_exp = random.sample(self.rb, k=self.batch_size)
        s_exp, a_exp, r_exp, s_next_exp, done_exp = zip(*sampled_exp)
        s_exp = torch.stack(s_exp)
        s_next_exp = torch.stack(s_next_exp)
        a_exp = torch.stack(a_exp)
        r_exp = torch.stack(r_exp)
        done_exp = torch.stack(done_exp)
        # compute q values for current and next states using dnn
        self.model.train()
        q_exp = self.model.forward(s_exp).gather(1, a_exp.view(-1, 1)).view(-1)
        with torch.no_grad():
            self.target_model.eval()
            q_exp_target = self.target_model.forward(s_next_exp).detach().max(1)[0]
        # compute mean loss of the batch
        loss = self.loss(q_exp, r_exp + self.gamma*q_exp_target*~done_exp)
        # compute gradient of loss
        self.agent.optimizer.zero_grad()
        loss.backward()
        # update weigths
        self.agent.optimizer.step()
        
        return loss.cpu().detach().numpy()

    def episode(self, ep, reset=False):
        if reset:
            self.agent.reset_env()
        if self.use_rb and ep == 0:
            print("Filling replay buffer before training...")
        # initialize starting state
        self.agent.reset_env()
        s = self.agent.detect_objects()
        s_old = s

        # iterate over timesteps
        loss_ep = 0
        ts_ep = 0
        r_ep = 0
        done = False
        while not done:
            self.ts_tot, ts_ep = self.ts_tot + 1, ts_ep + 1
            if self.target_model != self.model and (self.ts_tot % self.tm_wait) == 0:
                # update target model weigths as current self.model weights
                self.update_target()
            # Select action using the behaviour policy
            s_transf_cuboids = self.agent.transform_mask(s[0].copy())
            s_transf_borders = cv2.resize(s[1].copy(), (120, 92), interpolation=cv2.INTER_LINEAR)
            s_transf = np.stack([s_transf_cuboids, s_transf_borders])
            s_old_transf_cuboids = self.agent.transform_mask(s_old[0].copy())
            s_old_transf_borders = cv2.resize(s_old[1].copy(), (120, 92), interpolation=cv2.INTER_LINEAR)
            s_old_transf = np.stack([s_old_transf_cuboids, s_old_transf_borders])
            a = self.select_action(s_transf, s_old_transf)
            # Execute action in emulator and observe reward r and next state s_next
            s_next, r, done = self.agent.move(a.item())
            if done is None: done = False
            r_ep += r
            # transform s_next before storing it
            s_next_transf_cuboids = self.agent.transform_mask(s_next[0].copy())
            s_next_transf_borders = cv2.resize(s_next[1].copy(), (120, 92), interpolation=cv2.INTER_LINEAR)
            s_next_transf = np.stack([s_next_transf_cuboids, s_next_transf_borders])
            # add experience to replay buffer (as torch tensors)
            self.rb.append((torch.tensor(np.vstack([s_transf, s_old_transf])), a,
                torch.tensor(r, dtype=torch.float32),
                torch.tensor(np.vstack([s_next_transf, s_transf])),
                torch.tensor(done, dtype=torch.bool)))
            # set next state as the new current state
            s_old = s
            s = s_next
            # to fill the replay buffer before starting training
            if self.use_rb and len(self.rb) < self.batch_size:
                continue
            elif not self.training_started:
                self.training_started = True
                print("Training started")
            # execute a training step on the DQN
            loss_ep += self.training_step()
        
            # if (ep+1 % 1) == 0:
            print(f"[{ep+1}|{ts_ep}] Episode mean loss: {round(loss_ep/ts_ep, 4)} | Episode reward: {r_ep}")
        
        return loss_ep, r_ep

    def select_action(self, s, s_old):
        # Select a behaviour policy between epsilon-greedy and softmax (boltzmann)
        s = torch.tensor(np.vstack((s, s_old))).unsqueeze(0)
        with torch.no_grad():
            self.model.eval()
            q_values = self.model.forward(s)

        if self.policy == "egreedy":
            if self.epsilon is None:
                raise KeyError("Provide an epsilon")

            # annealing of epsilon
            if self.epsilon.__class__.__name__ == "tuple":  # linear annealing
                epsilon = self.epsilon[0] + (self.epsilon[1] - self.epsilon[0]) * (1 - self.ts_tot / self.epsilon[2])
            else:  # no annealing
                epsilon = self.epsilon
            # Randomly generate a value between [0,1] with a uniform distribution
            if np.random.uniform(0, 1) < epsilon:
                # Select random action
                a = torch.tensor(np.random.randint(0, len(self.agent.angles)), dtype=torch.int64)
            else:
                # Select most probable action
                a = argmax(q_values)
                
        elif self.policy == "softmax":
            if self.temp is None:
                raise KeyError("Provide a temperature")

            # we use the provided softmax function in Helper.py
            probs = softmax(q_values, torch.tensor([self.temp]))[0].cpu().detach().numpy()
            a = torch.tensor(np.random.choice(range(0, self.env.action_space.n), p=probs), dtype=torch.int64)
        else:
            exit("Please select an existent behaviour policy")
        
        return a