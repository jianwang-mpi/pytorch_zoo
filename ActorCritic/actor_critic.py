import gym
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
from itertools import count
env = gym.make('CartPole-v0')


class Net(nn.Module):
    def forward(self, x):
        x = self.affine1(x)
        policy_result = self.policy(self.activation(x))
        value_result = self.value(self.activation(x))
        return self.softmax(policy_result), value_result

    def __init__(self):
        super(Net, self).__init__()
        self.affine1 = nn.Linear(in_features=4, out_features=128)
        self.policy = nn.Linear(in_features=128, out_features=2)
        self.value = nn.Linear(in_features=128, out_features=1)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.saved_actions = []
        self.rewards = []

model = Net()
optimizer = optim.Adam(model.parameters())

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    state = Variable(state)
    policy_result, value_result = model(state)
    m = Categorical(policy_result)
    action = m.sample()
    model.saved_actions.append((m.log_prob(action), value_result))
    return action

def finish_episode():
    R=0
    policy_losses = []
    value_losses = []
    rewards = []
    for r in model.rewards[::-1]:
        R = r + 0.99 * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    for (log_prob, value), r in zip(model.saved_actions, rewards):
        reward = r - value.data[0, 0]
        policy_losses.append(- log_prob * reward)
        value_losses.append(F.smooth_l1_loss(value, Variable(torch.Tensor([r]))))

    optimizer.zero_grad()
    loss = torch.cat(policy_losses).sum() + torch.sum(torch.cat(value_losses))
    loss.backward()
    optimizer.step()
    model.rewards = []
    model.saved_actions = []

def main():
    running_reward = 10
    for ep in count(1):
        state = env.reset()
        for t in range(0, 10000):
            action = select_action(state)
            action = action.data[0]
            state, reward, done, _ = env.step(action)
            model.rewards.append(reward)
            if done:
                break
        running_reward = running_reward * 0.99 + t * 0.01
        finish_episode()
        if ep % 10 == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                ep, t, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()