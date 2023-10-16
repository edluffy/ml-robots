import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# https://arxiv.org/pdf/1509.02971.pdf

class DDPG():
    def __init__(self, state_dim, action_dim,
                 gamma=0.99, actor_alpha=3e-4, critic_alpha=3e-4, tau=0.001,
                 replay_size=1000000, batch_size=64, device='cuda'):
        self.gamma = gamma
        self.tau = tau

        self.actor = Actor(state_dim, action_dim).to(device)
        self.target_actor = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_alpha)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_alpha)

        self.replay_buffer = ReplayBuffer(state_dim, action_dim,
                                          replay_size, batch_size, device)

    def learn(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()
        target_q = self.target_critic(next_states, self.target_actor(next_states))
        target_q = rewards + (1-dones)*self.gamma*target_q
        q = self.critic(states, actions)

        # Update critic
        q_loss = F.mse_loss(q, target_q)
        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        p_loss = torch.mean(-self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        p_loss.backward()
        self.actor_optimizer.step()

        # Update targets
        self.soft_update(self.target_actor, self.actor)
        self.soft_update(self.target_critic, self.critic)

        return q_loss, p_loss

    def policy(self, state, max=1, train=True):
        action = self.actor(state)
        action *= max
        action += int(train)*torch.normal(0, 0.1*max, size=action.shape, device=action.device)
        return torch.clamp(action, -max, max)

    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.store(state, action, reward, next_state, done)

    def soft_update(self, target, source):
        for tp, p in zip(target.parameters(), source.parameters()):
            tp.data.copy_(p.data*self.tau + tp.data*(1-self.tau))

    def save(self, path):
        for model in ['actor', 'target_actor', 'critic', 'target_critic']:
            torch.save(getattr(self, model).state_dict(), '_'.join([path, 'ddpg', model]))

    def load(self, path):
        for model in ['actor', 'target_actor', 'critic', 'target_critic']:
            getattr(self, model).load_state_dict(torch.load('_'.join([path, 'ddpg', model])))

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

    def forward(self, s):
        a = F.relu(self.l1(s))
        a = F.relu(self.l2(a))
        a = torch.tanh(self.l3(a))
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, s, a):
        q = F.relu(self.l1(torch.cat([s, a], 1)))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q


class ReplayBuffer():
    def __init__(self, state_dim, action_dim, replay_size, batch_size, device):
        self.replay_size = replay_size
        self.batch_size = batch_size
        self.device = device
        self.cnt = 0
        self.states = torch.empty((replay_size, state_dim), device=device)
        self.actions = torch.empty((replay_size, action_dim), device=device)
        self.rewards = torch.empty((replay_size, 1), device=device)
        self.next_states = torch.empty((replay_size, state_dim), device=device)
        self.dones = torch.empty((replay_size, 1), device=device)

    def store(self, state, action, reward, next_state, done):
        self.cnt = self.cnt+1 if self.cnt < self.replay_size else 0
        self.states[self.cnt] = state
        self.actions[self.cnt] = action
        self.rewards[self.cnt] = reward
        self.next_states[self.cnt] = next_state
        self.dones[self.cnt] = done

    def random_index(self):
        return torch.randint(min(self.replay_size, self.cnt),
                             (self.batch_size,), device=self.device)

    def sample(self, idx=None):
        idx = idx or self.random_index()
        return self.states[idx], self.actions[idx], self.rewards[idx],\
               self.next_states[idx], self.dones[idx]
