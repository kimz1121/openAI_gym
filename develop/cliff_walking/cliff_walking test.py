import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

import time

import keras
from keras.models       import Sequential
from keras.layers       import Dense
from keras.optimizers   import Adam

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

# env = gym.make("LunarLander-v2", render_mode="human")
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
            MC 방법으로 접근 
            # incremental MC 방법으로 접근 
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

    
    문제
        
        1. 학습자체는 실행이 되나, 학습된 내용에 문제가 있음

        reward = reward_sum
        reward = reward_sum**2
        reward = reward_sum**5
        reward = reward**(eta*frame)
        등등의 방법은 항상 마지막 실패 상태에서 보상이 가장 크도록 첵정되어
        넘어지도록 유도하는 행동이 오히려 강하게 학습되는 경향이 발생한다.
        

        2. MC 방법과 Incremental MC 방법
            두 방법 모두 시도해 보앗으나, 
            완전한 렌덤 시나리오, 모델이 제어하는 시나리오 
            모두 학습이적절치 않음. 
            reward 자체에 문제가 크면 학습이 어려운 듯.




        
        
            

"""

def drive_env_random(env_arg, num_of_frame_arg, model_arg):
    observation, info = env_arg.reset()

    # check action and observation space size
    action_sapce_size = env_arg.env.action_space.n
    action_sapce = np.array(range(action_sapce_size))
    observation_sapce_size = env_arg.env.observation_space._shape[0]

    # create data space 
    x_input_rtn = np.zeros((num_of_frame_arg, observation_sapce_size+1))
    y_output_rtn = np.zeros((num_of_frame_arg, 1))

    #define temporal parameters
    reward_sum = 0
    frame = 0
    frame_set = 0
    
    #define hyper parameters
    gamma = 0.9#Q value discount
    alpha = 0.7#Q_function update ratio
    
    mode = 0 # 0 : SARSA, 1 : Q_learning

    # pick initial action
    action_value_all = perdict_model(model_arg, observation, action_sapce)
    action_index = action_value_all.argmax()
    action_pick = env_arg.action_space.sample()
    
    for i in range(num_of_frame_arg):
        frame += 1

        observation, reward, terminated, truncated, info = env_arg.step(action_pick)
        
        #pick next state and action ; which will be used in next iteration 
        action_value_all = perdict_model(model_arg, observation, action_sapce)
        action_index = action_value_all.argmax()
        action_pick = env_arg.action_space.sample()
        # action_value 중 최대 값을 선택하면 Q-learnig, action_value의 평균값을 사용하면 SARSA

        if mode == 0:
            action_value = action_value_all.mean()#SARSA mode
        else : 
            action_value = action_value_all[action_index]# Q_learning mode

        # if(reward_sum < 10):#reward seturation
        #     reward_sum += reward
        x_input_rtn[i, 0:observation_sapce_size] = observation[:] # 파이썬 인덱싱에 주의할 것. 인덱스가 1:4 이면 끝나는 인덱스가 0번 부터 3번까지
        x_input_rtn[i, observation_sapce_size:observation_sapce_size+1] = action_pick
        
        if terminated or truncated:
            if terminated == 1:
                reward = -100
            Q_target = reward

            observation, info = env_arg.reset()
            frame_set = i
            frame = 0
            reward_sum = 0

        else:
            Q_target = reward + gamma*action_value
        
        # print("=================")
        # print(Q_target)
        # print(y_output_rtn[i, :])
        y_output_rtn[i, :] = alpha*y_output_rtn[i, :] + (1-alpha)*Q_target
        # print(y_output_rtn[i, :])
        if i % 10 == 0:
            print("{}, {} and {} %".format(i, num_of_frame_arg, round((i/num_of_frame_arg)*100)))
        
        if terminated == 1:
            break

    return x_input_rtn, y_output_rtn



def drive_env_by_model(env_arg, num_of_frame_arg, model_arg):
    observation, info = env_arg.reset()

    # check action and observation space size
    action_sapce_size = env_arg.env.action_space.n
    action_sapce = np.array(range(action_sapce_size))
    observation_sapce_size = env_arg.env.observation_space._shape[0]

    # create data space 
    x_input_rtn = np.zeros((num_of_frame_arg, observation_sapce_size+1))
    y_output_rtn = np.zeros((num_of_frame_arg, 1))

    #define temporal parameters
    reward_sum = 0
    frame = 0
    frame_set = 0
    
    #define hyper parameters
    gamma = 0.9#Q value discount
    alpha = 0.7#Q_function update ratio
    
    mode = 0 # 0 : SARSA, 1 : Q_learning

    # pick initial action
    action_value_all = perdict_model(model_arg, observation, action_sapce)
    action_index = action_value_all.argmax()
    action_pick = action_sapce[action_index]
    
    for i in range(num_of_frame_arg):
        frame += 1

        observation, reward, terminated, truncated, info = env_arg.step(action_pick)
        
        #pick next state and action ; which will be used in next iteration 
        action_value_all = perdict_model(model_arg, observation, action_sapce)
        # print("=======================")
        # print(action_sapce)
        # print(action_value_all)
        action_index = action_value_all.argmax()
        action_pick = action_sapce[action_index]
        # action_value 중 최대 값을 선택하면 Q-learnig, action_value의 평균값을 사용하면 SARSA

        if mode == 0:
            action_value = action_value_all.mean()#SARSA mode
        else : 
            action_value = action_value_all[action_index]# Q_learning mode

        # if(reward_sum < 10):#reward seturation
        #     reward_sum += reward
        x_input_rtn[i, 0:observation_sapce_size] = observation[:] # 파이썬 인덱싱에 주의할 것. 인덱스가 1:4 이면 끝나는 인덱스가 0번 부터 3번까지
        x_input_rtn[i, observation_sapce_size:observation_sapce_size+1] = action_pick
        
        if terminated or truncated: 
            if terminated == 1:
                reward = -100
            Q_target = reward

            observation, info = env_arg.reset()
            frame_set = i
            frame = 0
            reward_sum = 0

        else:
            Q_target = reward + gamma*action_value

        # print(Q_target)
        y_output_rtn[i, :] = alpha*y_output_rtn[i, :] + (1-alpha)*Q_target

        if i % 10 == 0:
            print("{}, {} and {} %".format(i, num_of_frame_arg, round((i/num_of_frame_arg)*100)))
        
        if terminated == 1:
            break
    return x_input_rtn, y_output_rtn



def create_model(env_arg):
    observation_sapce_size = env_arg.env.observation_space._shape[0]

    model_rtn = Sequential()
    model_rtn.add(keras.Input(shape=(observation_sapce_size+1)))
    model_rtn.add(Dense(128, activation='sigmoid'))
    model_rtn.add(Dense(64, activation='sigmoid'))
    model_rtn.add(Dense(16, activation='sigmoid'))
    model_rtn.add(Dense(1, activation='sigmoid'))
    model_rtn.compile(loss='mse', optimizer=Adam())
    
    return model_rtn

def create_model_sigmoid(env_arg):
    observation_sapce_size = env_arg.env.observation_space._shape[0]

    model_rtn = Sequential()
    model_rtn.add(keras.Input(shape=(observation_sapce_size+1)))
    model_rtn.add(Dense(128, activation='sigmoid'))
    model_rtn.add(Dense(64, activation='sigmoid'))
    model_rtn.add(Dense(16, activation='sigmoid'))
    model_rtn.add(Dense(1, activation='sigmoid'))
    model_rtn.compile(loss='mse', optimizer=Adam())
    
    return model_rtn

def create_model_relu(env_arg):
    observation_sapce_size = env_arg.env.observation_space._shape[0]

    model_rtn = Sequential()
    model_rtn.add(keras.Input(shape=(observation_sapce_size+1)))
    model_rtn.add(Dense(64, activation='relu'))
    model_rtn.add(Dense(64, activation='relu'))
    model_rtn.add(Dense(1, activation='relu'))
    model_rtn.compile(loss='mse', optimizer=Adam())
    
    return model_rtn

def learning_model(model_arg, x_input_arg, y_output_arg):
    model_arg.fit(x_input_arg, y_output_arg, batch_size=100, epochs = 2)


def learning_model_batch_10(model_arg, x_input_arg, y_output_arg):
    model_arg.fit(x_input_arg, y_output_arg, batch_size=10, epochs = 2)


def perdict_model(model_arg, observation_arg, action_sapce_arg):
    # it retruns action values at observation
    # model_arg : keras model
    # observation_arg : np.array (1, 8)
    # action_sapce_arg : np.array (1, 4) 
    
    action_space_size = action_sapce_arg.shape[0]
    observation_space_size = observation_arg.shape[0]

    action_value_rtn = np.empty((action_space_size))
    model_input = np.empty((1, observation_space_size + 1))
    
    # print(model_input.shape)
    for i in range(action_space_size):#action_space_arg의 열길이에 따라 평가를 반복
        action = action_sapce_arg[i]
        # print("================")
        # print(observation_arg)
        # print(action)
        model_input[0, 0:observation_space_size] = observation_arg  # 0~3번 인덱스
        model_input[0, observation_space_size:observation_space_size+1] = action             #4번 인덱스
        # print(model_input)
        # print(model_input.shape)
        
        action_value_rtn[i] = model_arg.predict(model_input, verbose = 0) # model input은 "1차원 길이 (9)" 형태가 아닌 2차원 크기 (1, 9) 로 사용하여야 한다. 

    return action_value_rtn

def pick_action(model_arg, observation_arg, action_sapce_arg):
    action_value = perdict_model(model_arg, observation_arg, action_sapce_arg)
    action_index = action_value.argmax()
    action_pick = action_sapce_arg[action_index]

    return action_pick

def pick_action_verbose(model_arg, observation_arg, action_sapce_arg):
    action_value = perdict_model(model_arg, observation_arg, action_sapce_arg)
    action_index = action_value.argmax()
    action_pick = action_sapce_arg[action_index]

    print("=================")
    print(type(action_value))
    print(action_value)
    print(type(action_index))
    print(action_index)
    print(type(action_pick))
    print(action_pick)

    return action_pick

if __name__ == '__main__':
    # env = gym.make("CartPole-v1", render_mode="human")
    env =  gym.make ("CliffWalking-v0")

    print("================")
    print(env.action_space)
    print(env.action_space.n)
    print(type(env.action_space.n))


    print(env.observation_space._shape)
    print(env.observation_space._shape[0])
    print(type(env.observation_space._shape[0]))

    num_of_frame = 100
    model_glb = create_model_sigmoid(env)

    #initial random drive
    for i in range(1000):
        x_input, y_output = drive_env_random(env, num_of_frame, model_glb)
        learning_model(model_glb, x_input, y_output)
    
    env_screen = gym.make("CliffWalking-v0", render_mode="human")
    #incremental MC 
    num_of_frame = 1000
    iter = 1000
    for i in range(iter):
    #     print("{}/{}======================================".format(i, iter))
        
        # if(i%2 == 0):
        if(i % 5 == 0):
            x_input, y_output = drive_env_random(env_screen, num_of_frame, model_glb)
            print("{}/{}======================================".format(i, iter))
            learning_model_batch_10(model_glb, x_input, y_output)
        else:
            x_input, y_output = drive_env_by_model(env_screen, num_of_frame, model_glb)
            print("{}/{}======================================".format(i, iter))
            learning_model(model_glb, x_input, y_output)

        # x_input, y_output = drive_env_by_model(env_screen, num_of_frame, model_glb)
        # print("{}/{}======================================".format(i, iter))
        # learning_model_batch_10(model_glb, x_input, y_output)

    num_of_frame = 100000
    x_input, y_output = drive_env_by_model(env_screen, num_of_frame, model_glb)
    # learning_model_batch_10(model_glb, x_input, y_output)
    env_screen.close()
    env.close()