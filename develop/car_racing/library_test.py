import gymnasium as gym
env = gym.make("CarRacing-v2", render_mode="human")
# env = gym.make("LunarLander")
observation, info = env.reset()

for _ in range(1500):
    # env.render()
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    print(_)

    if terminated or truncated:
        observation, info = env.reset()

env.close()