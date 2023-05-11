import gymnasium as gym
import matplotlib.pyplot as plt
env = gym.make("LunarLander-v2", render_mode="human")
# env = gym.make("LunarLander-v2", render_mode="rgb_array") 
# # WSL2 에서 pygame 화면 출력이 안되던 문제는, anaconda 가상환경의 화면 출력 라이브러리 관련 문재 였음. 
# 현재는 가상환경 사용을 배제 하거나, 가상환경용 라이브러리 폴더에 필요한 라이브러리 파일을 리눅스 라이브러리에서 복사해 옴으로써 해결 

observation, info = env.reset()

for _ in range(1000):    
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    print(_)
    print(observation)

    if terminated or truncated:
        observation, info = env.reset()

env.close()