import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


env = gym.make("LunarLander-v2", render_mode="human")
# env = gym.make("LunarLander-v2", render_mode="rgb_array") 
# # WSL2 에서 pygame 화면 출력이 안되던 문제는, anaconda 가상환경의 화면 출력 라이브러리 관련 문재 였음. 
# 현재는 가상환경 사용을 배제 하거나, 가상환경용 라이브러리 폴더에 필요한 라이브러리 파일을 리눅스 라이브러리에서 복사해 옴으로써 해결 

"""
문제 분석

    문제 : LunarLander-v2

    특성 : 
        보상이 희박하다는 특성이 있음
        각 시나리오마다 상황이 종료되고 나서야 문제를 해결할 수 있음.

        일정한 바람, 난류 등, 외란이 있음



풀이 전략
    방법
        신경망 응용 강화학습

        함수 평가
            incremental MC 방법으로 접근 
                장점 가장 간단한 방법

        네트워크 설계
            기존에 존재하는 손실함수 그대로 사용하기 위해
            비효율적이지만 가장 고전적인 방법으로 접근. 
            
            네트워크 입력
                상황, 행동
            네트워크 출력
                리워드

    학습결과 실행
        상황과 행동에 따라 리워드를 예측하는 함수가 학습하였으니,
        각 상황마다. 4가지 행동을 대입하여 그중 무엇이 가장 효과적인지 비교후 최고를 선택하는 방법적용.
            

"""


observation, info = env.reset()# 초기 상태에 관한 정보.

print(observation)
print(type(observation))
print(observation.shape)

num_of_frame = 1000
x_input = np.empty((num_of_frame, 8+1))
y_output = np.empty((num_of_frame, 1))


for i in range(num_of_frame):    
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    # reward는 프래임 단위로 주어진다.
    # 한 프래임의 action에 대하여 하나의 reward를 주는 방식. 

    print(action)
    print(type(action))
    print(observation.shape)

    x_input[i, 0:8] = observation[:] # 파이썬 인덱싱에 주의할 것. 인덱스가 1:8 이면 끝나는 인덱스가 0번 부터 7번까지
    x_input[i, 8] = action
    y_output[i, :] = reward
    print(i)
    print(reward)
    print(type(reward))

    if terminated or truncated:
        observation, info = env.reset()


print(x_input.shape)
print(y_output.shape)

print(x_input[99, :])
print(y_output[99, :])



env.close()