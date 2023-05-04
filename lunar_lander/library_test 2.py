import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

import time

import keras
from keras.models       import Sequential
from keras.layers       import Dense
from keras.optimizers   import Adam

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
            

"""

def drive_env_random(env_arg, num_of_frame_arg):

    observation, info = env_arg.reset()# 초기 상태에 관한 정보.

    print(observation)
    print(type(observation))
    print(observation.shape)

    x_input_rtn = np.empty((num_of_frame_arg, 8+1))
    y_output_rtn = np.empty((num_of_frame_arg, 1))


    for i in range(num_of_frame_arg):    
        action = env_arg.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env_arg.step(action)
        # reward는 프래임 단위로 주어진다.
        # 한 프래임의 action에 대하여 하나의 reward를 주는 방식. 

        print(action)
        print(type(action))
        print(observation.shape)

        x_input_rtn[i, 0:8] = observation[:] # 파이썬 인덱싱에 주의할 것. 인덱스가 1:8 이면 끝나는 인덱스가 0번 부터 7번까지
        x_input_rtn[i, 8] = action
        y_output_rtn[i, :] = reward
        print(i)
        print(reward)
        print(type(reward))

        if terminated or truncated: 
            # 시뮬레이터가 종료되면 재시작이 필요
            # 조건에 따라 상황이 종료되면, 쓸모있는 데이터를 모을 것으로 기대되는 상황이 더이상 없다고 취급한다.
            observation, info = env_arg.reset()


    print(x_input_rtn.shape)
    print(y_output_rtn.shape)

    print(x_input_rtn[num_of_frame_arg-1, :])
    print(y_output_rtn[num_of_frame_arg-1, :])

    return x_input_rtn, y_output_rtn

def create_model():
    model_rtn = Sequential()
    model_rtn.add(keras.Input(shape=(9,)))
    model_rtn.add(Dense(128, activation='sigmoid'))
    model_rtn.add(Dense(64, activation='sigmoid'))
    model_rtn.add(Dense(1, activation='sigmoid'))
    model_rtn.compile(loss='mse', optimizer=Adam())
    
    return model_rtn

def learning_model(model_arg, x_input_arg, y_output_arg):
    model_arg.fit(x_input_arg, y_output_arg, batch_size=10, epochs = 5)

def perdict_model(model_arg, observation_arg, action_sapce_arg):
    # it retruns action values at observation
    # model_arg : keras model
    # observation_arg : np.array (1, 8)
    # action_sapce_arg : np.array (1, 4) 
    action_space_column_len = action_sapce_arg.shape[0]
    observation_space_column_len = observation_arg.shape[0]

    action_value_rtn = np.empty((action_space_column_len))
    model_input = np.empty((1, observation_space_column_len + 1))
    
    print(model_input.shape)
    for i in range(action_space_column_len):#action_space_arg의 열길이에 따라 평가를 반복
        action = action_sapce_arg[i]
        print("================")
        print(observation_arg)
        print(action)
        model_input[0, 0:8] = observation_arg  # 0~7번 인덱스
        model_input[0, 8] = action             #8번 인덱스
        print(model_input)
        print(model_input.shape)
        
        action_value_rtn[i] = model_arg.predict(model_input) # model input은 "1차원 길이 (9)" 형태가 아닌 2차원 크기 (1, 9) 로 사용하여야 한다. 

    return action_value_rtn

if __name__ == '__main__':
    env = gym.make("LunarLander-v2", render_mode="human")

    num_of_frame = 100
    x_input, y_output = drive_env_random(env, num_of_frame)
    print(x_input.shape)
    print(y_output.shape)

    model_glb = create_model()
    learning_model(model_glb, x_input, y_output)
    
    # test_obs = np.array([ 0.84823453  0.26283216  1.11286175 -1.18096614 -0.27859148 -0.07228909,  0.          0.          0.        ])
    test_obs = np.array([0.84823453, 0.26283216, 1.11286175, -1.18096614, -0.27859148, -0.07228909, 0, 0])

    model_glb.summary()
    # keras.utils.plot_model(model_glb, "my_first_model_with_shape_info.png", show_shapes=True)


    action_space = np.array([0, 1, 2, 3])
    # print(action_space.shape)
    action_value = perdict_model(model_glb, test_obs, action_space)
    # print(action_value)

    env.close()