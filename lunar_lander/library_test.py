import gymnasium as gym
import matplotlib.pyplot as plt
env = gym.make("LunarLander-v2", render_mode="human")
# env = gym.make("LunarLander-v2", render_mode="rgb_array")
# env = gym.make("LunarLander-v2")
observation, info = env.reset()

print(env.observation_space.shape)

# screen = env.render()
# print(screen)

# plt.imshow(screen[0])

env.close()


# for _ in range(1000):
    
#     action = env.action_space.sample()  # agent policy that uses the observation and info
#     observation, reward, terminated, truncated, info = env.step(action)

#     print(observation)

#     print(_)

#     if terminated or truncated:
#         observation, info = env.reset()

# env.close()