import gymnasium as gym
from agents.ddpg import DDPG
import torch

device = 'cuda'
epochs = 100

env = gym.make('Pendulum-v1', g=9.81, render_mode="human")

agent = DDPG(
    state_dim=3,
    action_dim=1,
    actor_alpha=0.002,
    critic_alpha=0.001,
    tau=0.005,
    replay_size=50000,
    batch_size=64,
    device = device,
)

if __name__ == "__main__":
    for epoch in range(epochs):
        prev_state, _ = env.reset()
        prev_state = torch.tensor(prev_state, device=device)

        done = False
        ep_reward = 0
        while not done:
            action = agent.policy(prev_state, max=2).detach().cpu().numpy()
            state, reward, done, trunc, _ = env.step(action)

            state = torch.tensor(state, device=device)
            action = torch.tensor(action, device=device)
            agent.store(prev_state, action, reward, state, done)

            q_loss, p_loss = agent.learn()
            ep_reward += reward

            prev_state = state

            if trunc:
                break
        print('epoch:', epoch, ' reward:', ep_reward)

    env.close()
