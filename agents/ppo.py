import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

# https://arxiv.org/pdf/1707.06347.pdf

class PPO():
    def __init__(self, state_dim, action_dim, env_dim, nsteps,
                 actor_alpha=1e-2, critic_alpha=1e-2, gamma=0.999,
                 gae_lambda=0.95, clip_ratio=0.2, target_kl=1e-2,
                 actor_iters=10, critic_iters=10, device='cuda'):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.actor_iters = actor_iters
        self.critic_iters = critic_iters

        self.actor = Actor(state_dim, action_dim, device)
        self.actor_optimizer = optim.Adam(list(
            self.actor.parameters())+[self.actor.log_sd], lr=actor_alpha)

        self.critic = Critic(state_dim, device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_alpha)

        self.memory = Memory(state_dim, action_dim, env_dim, nsteps, device)
        self.store = self.memory.store

    def learn(self, last_state):
        p_losses, v_losses, approx_kls, clip_fracs = [], [], [], []
        states, actions, rewards, dones, log_probs = self.memory.fetch()
        states[-1] = last_state

        v_last = self.critic(states[-1]).detach()
        rewards[-1] += self.gamma*(1-dones[-1])*v_last
        returns = self.get_returns(rewards, dones)

        # Critic learn
        for _ in range(self.critic_iters):
            v_pred = self.critic(states)
            v_loss = torch.mean((v_pred[:-1]-returns)**2)
            self.critic_optimizer.zero_grad()
            v_loss.backward()
            self.critic_optimizer.step()

            v_losses.append(v_loss)

        v_pred = v_pred.detach()
        td_errors = rewards + self.gamma*(1-dones)*v_pred[1:] - v_pred[:-1]
        advantages = self.get_advantages(td_errors)

        # Actor learn
        for x in range(self.actor_iters):
            new_log_probs = self.actor(states[:-1]).log_prob(actions)
            r_theta = torch.exp(new_log_probs-log_probs)
            clip_frac = torch.where(advantages > 0,
                                        torch.min(r_theta, torch.tensor((1+self.clip_ratio))),
                                        torch.max(r_theta, torch.tensor((1-self.clip_ratio))),
                            )

            p_loss = torch.mean(-clip_frac*advantages)
            self.actor_optimizer.zero_grad()
            p_loss.backward()
            self.actor_optimizer.step()

            approx_kl = (log_probs-new_log_probs).mean().item()
            #if approx_kl > self.target_kl:
            #    break

            clip_fracs.append(clip_frac)
            approx_kls.append(approx_kl)
            p_losses.append(p_loss)

        return p_losses, v_losses, approx_kls, clip_fracs

    def policy(self, state, max=1, train=True):
        dist = self.actor(state)
        action = dist.sample() if train else dist.mean
        log_prob = dist.log_prob(action).detach()
        return action, log_prob

    # Back-sample through rewards for discounted returns at each time step
    def get_returns(self, rewards, dones):
        returns = torch.empty_like(rewards)
        returns[-1] = rewards[-1]
        for t in range(len(rewards)-2, -1, -1):
            returns[t] = self.gamma*(1-dones[t])*returns[t+1] + rewards[t]
        return returns

    # Back-sample through td_errors for advantages at each time step
    def get_advantages(self, td_errors):
        advantages = torch.empty_like(td_errors)
        advantages[-1] = td_errors[-1]
        for t in range(len(td_errors)-2, -1, -1):
            advantages[t] = self.gamma*self.gae_lambda*advantages[t+1] + td_errors[t]
        advantages = (advantages - advantages.mean()) / advantages.std()
        return advantages

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super().__init__()
        self.l1 = torch.nn.Linear(state_dim, 64)
        self.l2 = torch.nn.Linear(64, 64)
        self.l3 = torch.nn.Linear(64, action_dim)
        self.log_sd = torch.ones(action_dim, device=device, requires_grad=True)
        self.to(device)

    def forward(self, s):
        mean = torch.tanh(self.l1(s))
        mean = torch.tanh(self.l2(mean))
        mean = self.l3(mean)
        sd = torch.diag(torch.exp(self.log_sd))
        return MultivariateNormal(mean, sd)


class Critic(nn.Module):
    def __init__(self, state_dim, device):
        super().__init__()
        self.l1 = torch.nn.Linear(state_dim, 64)
        self.l2 = torch.nn.Linear(64, 64)
        self.l3 = torch.nn.Linear(64, 1)

        with torch.no_grad():
            self.l3.weight.fill_(0)
            self.l3.bias.fill_(0)
        self.to(device)

    def forward(self, s):
        v = torch.tanh(self.l1(s))
        v = torch.tanh(self.l2(v))
        v = self.l3(v)
        return v.squeeze()


class Memory():
    def __init__(self, state_dim, action_dim, env_dim, nsteps, device):
        self.step = 0
        self.states = torch.empty((nsteps+1, env_dim, state_dim), device=device)
        self.actions = torch.empty((nsteps, env_dim, action_dim), device=device)
        self.rewards = torch.empty((nsteps, env_dim), device=device)
        self.dones = torch.empty((nsteps, env_dim), device=device)
        self.log_probs = torch.empty((nsteps, env_dim), device=device)

    def store(self, states, actions, rewards, dones, log_probs):
        self.states[self.step] = states
        self.actions[self.step] = actions
        self.rewards[self.step] = rewards
        self.dones[self.step] = dones
        self.log_probs[self.step] = log_probs
        self.step += 1

    def fetch(self):
        self.step = 0
        return self.states, self.actions, self.rewards, self.dones, self.log_probs
