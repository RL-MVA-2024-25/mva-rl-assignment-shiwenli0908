from copy import deepcopy

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from gymnasium.wrappers import TimeLimit

from env_hiv import HIVPatient
from evaluate import evaluate_HIV, evaluate_HIV_population

import os

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity) # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)

def greedy_action(network, state):
    device = "cuda" if next(network.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()

class DQNModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super(DQNModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.model(x)

class ProjectAgent:
    def __init__(self):
        
        self.nb_actions = 4
        self.nb_states = 6
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQNModel(self.nb_states, self.nb_actions, hidden_dim=512).to(self.device)
        self.gamma = 0.95
        self.batch_size = 40
        buffer_size = 20000
        self.memory = ReplayBuffer(buffer_size, self.device)
        self.epsilon_max = 1.
        self.epsilon_min = 0.01
        self.epsilon_stop = 10000
        self.epsilon_delay = 2000
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        
        self.target_model = deepcopy(self.model).to(self.device)
        self.criterion = torch.nn.SmoothL1Loss()
        lr = 0.001
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.nb_gradient_steps = 1
        self.update_target_strategy = 'replace'
        self.update_target_freq = 20
        self.update_target_tau = 0.005

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
    
    def act(self, observation, use_random=False):
        """Select an action based on the current observation."""
        if use_random:
            return np.random.randint(0, self.nb_actions)
        with torch.no_grad():
            state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.model(state).argmax(dim=1).item())

    def save(self, path):
        """Save the model parameters to a file."""
        torch.save(self.model.state_dict(), path)

    def load(self, path="./dqn_model_best.pth"):
        """Load the model parameters from a file."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

    def train(self, env, max_episode):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        best_score = -1  # 初始化最佳分数

        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)
            # next transition
            step += 1
            if done or trunc:
                episode += 1
                print("Episode ", '{:3d}'.format(episode), 
                      ", epsilon ", '{:6.2f}'.format(epsilon), 
                      ", batch size ", '{:5d}'.format(len(self.memory)), 
                      ", episode return ", '{:4.1f}'.format(episode_cum_reward),
                      sep='')

                if episode > 150:
                    # Evaluate the agent's performance
                    score_agent = evaluate_HIV(agent=self, nb_episode=5)
                    print(f"Evaluation Score: {score_agent}")
                    if score_agent > best_score:
                        best_score = score_agent
                        self.save("dqn_model_best.pth")
                        print(f"New best score {best_score} achieved. Model saved.")

                state, _ = env.reset()
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state
        return episode_return

if __name__ == "__main__":

    env = TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=200)  # The time wrapper limits the number of steps in an episode at 200.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train agent
    agent = ProjectAgent()
    #agent.load()

    scores = agent.train(env, 300)
    #agent.save("dqn_model_rl2.pth")















