import gym
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
from collections import namedtuple
from itertools import count
import random
import math
from copy import deepcopy
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
env = gym.make('CartPole-v0')
env = env.unwrapped

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, 2)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


net = Net()

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class Memory:
    def __init__(self, length):
        self.memory = np.array()
        self.position = 0
        self.length = length

    def __len__(self):
        return len(self.memory)

    def push(self, state, action, next_state, reward):
        """Saves a transition."""
        if len(self.memory) < self.length:
            self.memory.append(None)
        state = torch.from_numpy(state).float()
        next_state = torch.from_numpy(next_state).float()
        reward = torch.FloatTensor([reward])
        self.memory[self.position] = Transition(state, action, next_state, reward)
        self.position = (self.position + 1) % self.length

    def clear(self):
        del self.memory[:]

    def sample(self, batch_size):
        sample_index = np.random.choice(len(self.memory), batch_size)
        b_memory = self.memory[sample_index, :]
        return b_memory

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    state = Variable(state, volatile=True)
    # if torch.cuda.is_available():
    #     state = state.cuda()
    value_result = net(state)
    sample = random.random()
    if sample < EPS_START:
        return value_result.data.max(1)[1].view(1, 1)
    else:
        return torch.LongTensor([[random.randrange(2)]])


optimizer = optim.Adam(params=net.parameters(), lr=0.01)


def finish_episode(memory):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # We don't want to backprop through the expected action values and volatile
    # will save us on temporarily changing the model parameters'
    # requires_grad to False!
    next_states = Variable(torch.cat([s for s in batch.next_state
                                                if s is not None]),
                                     volatile=True)
    state_batch = Variable(batch.state, volatile=True)
    action_batch = Variable(batch.action)
    reward_batch = Variable(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = net(next_states).max(1)[0].view(128, 1)
    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    next_state_values.volatile = False
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch.view(128, 1)

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def main():
    running_reward = 10
    memory = Memory(1000)
    for ep in count(1):
        state = env.reset()
        for t in range(0, 10000):
            action = select_action(state)
            next_state, reward, done, _ = env.step(action[0, 0])
            # modify the reward

            x, x_dot, theta, theta_dot = next_state
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2
            memory.push(state, action, next_state, r)

            if done:
                break
            state = next_state
        running_reward = running_reward * 0.99 + t * 0.01
        finish_episode(memory)
        if ep % 10 == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                ep, t, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()
