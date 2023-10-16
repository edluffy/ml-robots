import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# https://arxiv.org/pdf/1706.02275.pdf

class MADDPG():
    def __init__(self, agents, state_dims, action_dims,
                 gamma=0.95, actor_alpha=0.01, critic_alpha=0.01, tau=0.01,
                 replay_size=1000000, batch_size=1024, device='cuda'):
        self.agents = agents
        self.gamma = gamma
        self.tau = tau

        self.actor, self.target_actor, self.actor_optimizer = {}, {}, {}
        self.critic, self.target_critic, self.critic_optimizer = {}, {}, {}
        self.replay_buffer = {}

        for agent in self.agents:
            # Decentralised actors
            self.actor[agent] = Actor(state_dims[agent], action_dims[agent]).to(device)
            self.target_actor[agent] = copy.deepcopy(self.actor[agent])
            self.actor_optimizer[agent] = optim.Adam(self.actor[agent].parameters(), lr=actor_alpha)
            # Centralised critics
            self.critic[agent] = Critic(sum(state_dims.values()), sum(action_dims.values())).to(device)
            self.target_critic[agent] = copy.deepcopy(self.critic[agent])
            self.critic_optimizer[agent] = optim.Adam(self.critic[agent].parameters(), lr=critic_alpha)

            self.replay_buffer[agent] = ReplayBuffer(state_dims[agent], action_dims[agent],
                                            replay_size, batch_size, device)

    def learn(self):
        idx, ready = self.replay_buffer[self.agents[0]].random_index()
        if not ready: return {}, {}
        batch = {a: self.replay_buffer[a].sample(idx) for a in self.agents}

        # Centralised data
        multi_states = torch.cat([batch[a][0] for a in self.agents], -1)
        multi_next_states = torch.cat([batch[a][3] for a in self.agents], -1)
        multi_old_actions = torch.cat([batch[a][1] for a in self.agents], -1)
        multi_actions = torch.cat([self.actor[a](batch[a][0]) for a in self.agents], -1)
        multi_target_next_actions = torch.cat([self.target_actor[a](batch[a][3]) for a in self.agents], -1)

        q_loss, p_loss = {}, {}
        for agent in self.agents:
            states, actions, rewards, next_states, dones = batch[agent]
            target_q = self.target_critic[agent](multi_next_states, multi_target_next_actions)
            target_q = rewards + (1-dones)*self.gamma*target_q
            q = self.critic[agent](multi_states, multi_old_actions)

            # Update critic
            q_loss[agent] = F.mse_loss(q, target_q)
            self.critic_optimizer[agent].zero_grad()
            q_loss[agent].backward(retain_graph=True)
            self.critic_optimizer[agent].step()

            # Update actor
            p_loss[agent] = torch.mean(-self.critic[agent](multi_states, multi_actions))
            self.actor_optimizer[agent].zero_grad()
            p_loss[agent].backward(retain_graph=True)
            self.actor_optimizer[agent].step()

            # Update targets
            self.soft_update(self.target_actor[agent], self.actor[agent])
            self.soft_update(self.target_critic[agent], self.critic[agent])

        return q_loss, p_loss

    def policy(self, state, max=1, train=True):
        action = {}
        for agent in self.agents:
            action[agent] = self.actor[agent](torch.tensor(state[agent], device="cuda"))
            action[agent] *= max
            action[agent] += int(train)*torch.normal(0, 0.1*max, size=action[agent].shape,
                                            device=action[agent].device)
            action[agent] = torch.clamp(action[agent], -max, max)
            action[agent] = action[agent].squeeze()
        return action

    def store(self, state, action, reward, next_state, done):
        for agent in self.agents:
            self.replay_buffer[agent].store(state[agent], action[agent], reward[agent],
                                            next_state[agent], done[agent])

    def soft_update(self, target, source):
        for tp, p in zip(target.parameters(), source.parameters()):
            tp.data.copy_(p.data*self.tau + tp.data*(1-self.tau))

    def save(self, path):
        for agent in self.agents:
            for model in ['actor', 'target_actor', 'critic', 'target_critic']:
                file = '_'.join([path, 'maddpg', agent, model])
                torch.save(getattr(self, model)[agent].state_dict(), file)

    def load(self, path):
        for agent in self.agents:
            for model in ['actor', 'target_actor', 'critic', 'target_critic']:
                file = '_'.join([path, 'maddpg', agent, model])
                getattr(self, model)[agent].load_state_dict(torch.load(file))


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, action_dim)

    def forward(self, s):
        a = F.relu(self.l1(s))
        a = F.relu(self.l2(a))
        a = torch.tanh(self.l3(a))
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 1)

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
        idx = self.cnt % self.replay_size
        self.states[idx] = torch.tensor(state, device=self.device)
        self.actions[idx] = torch.tensor(action, device=self.device)
        self.rewards[idx] = torch.tensor(reward, device=self.device)
        self.next_states[idx] = torch.tensor(next_state, device=self.device)
        self.dones[idx] = torch.tensor(done, device=self.device)
        self.cnt += 1

    def sample(self, idx=None):
        if idx is None: idx, _ = self.random_index()
        return self.states[idx], self.actions[idx], self.rewards[idx],\
               self.next_states[idx], self.dones[idx]

    def random_index(self):
        return torch.randint(min(self.replay_size, self.cnt),
                             (self.batch_size,), device=self.device), self.cnt >= self.batch_size
