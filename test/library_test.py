import gymnasium as gym
env = gym.make("Hopper-v4", render_mode="human")
observation, info = env.reset()

for _ in range(100):
    env.render()
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    print(_)

    if terminated or truncated:
        observation, info = env.reset()

env.close()