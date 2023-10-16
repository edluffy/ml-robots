from pettingzoo.mpe import simple_tag_v3
from agents.maddpg import MADDPG
import numpy as np
import torch

device = 'cuda'
epochs = 30000
nsteps = 4

env = simple_tag_v3.parallel_env(
    #render_mode="human",
    continuous_actions=True,
    max_cycles=100,
)

multi_agent = MADDPG(
    agents = env.possible_agents,
    state_dims={k: v.shape[0] for k, v in env.observation_spaces.items()},
    action_dims={k: v.shape[0] for k, v in env.action_spaces.items()},
    actor_alpha=1e-4,
    critic_alpha=1e-3,
    tau=0.001,
    replay_size=100000,
    batch_size=32,
    device = device,
)

if __name__ == '__main__':
    total_steps = 0
    ep_rewards = [0] * epochs
    q_loss, p_loss = {}, {}

    multi_agent.load('models/test_maddpg_simple_tag')

    for epoch in range(epochs):
        prev_state, _ = env.reset()
        for agent in env.possible_agents:
            prev_state[agent] = torch.tensor(prev_state[agent], device='cuda')

        for _ in range(100):
            action = multi_agent.policy(prev_state, max=0.5)
            for agent in env.possible_agents:
                action[agent] = action[agent].detach().cpu().numpy() + 0.5

            state, reward, _, done, _ = env.step(action)
            for agent in env.possible_agents:
                state[agent] = torch.tensor(state[agent], device=device)
                action[agent] = torch.tensor(action[agent], device=device)
            multi_agent.store(prev_state, action, reward, state, done)

            if total_steps % nsteps == 0:
                q_loss, p_loss = multi_agent.learn()

            ep_rewards[epoch] += sum(reward.values())
            prev_state = state
            total_steps += 1

        if epoch % 100 == 0:
            print('epoch:', epoch, ' ave. reward:', np.mean(ep_rewards[epoch-100:epoch]))
            multi_agent.save('models/test_maddpg_simple_tag')
    env.close()

