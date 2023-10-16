from hide_and_seek_robot.nav_env import NavEnv
from agents.maddpg import MADDPG

device = 'cuda'
epochs = 10000
max_cycles = 1000

env = NavEnv(
    show_viewer=True,
    max_cycles=max_cycles,
)

multi_agent = MADDPG(
    agents=env.agents_list,
    state_dims=env.state_dims,
    action_dims=env.action_dims,
    actor_alpha=0.01,
    critic_alpha=0.01,
    tau=0.01,
    replay_size=1000000,
    batch_size=1024,
    device = device,
)


if __name__ == '__main__':
    total_steps = 0
    ep_rewards = [0] * epochs
    q_loss, p_loss = {}, {}

    for epoch in range(epochs):
        prev_state, _ = env.reset()
        for step in range(max_cycles):
            action = multi_agent.policy(prev_state)

            state, reward, done, _ = env.step(action)
            multi_agent.store(prev_state, action, reward, state, done)

            if total_steps % 100 == 0:
                q_loss, p_loss = multi_agent.learn()

            ep_rewards[epoch] += sum(reward.values())
            prev_state = state
            total_steps += 1

        print('epoch:', epoch, 'rewards:', ep_rewards[epoch])

    env.close()
