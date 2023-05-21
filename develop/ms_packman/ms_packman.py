import gymnasium as gym
import matplotlib.pyplot as plt
env = gym.make("MsPacman-v4", render_mode="human")
# env = gym.make("MsPacman-v0", render_mode="rgb_array")
# env = gym.make("MsPacman-v0")
observation, info = env.reset()

# screen = env.render()
# print(screen)
# env.close()

# plt.imshow(screen[0])



for _ in range(1000):
    
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    print(observation)

    print(_)

    if terminated or truncated:
        observation, info = env.reset()

env.close()