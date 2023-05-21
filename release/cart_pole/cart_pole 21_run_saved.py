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

def drive_env_random(env_arg, num_of_frame_arg, model_arg):
    observation, info = env_arg.reset()

    # check action and observation space size
    action_sapce_size = env_arg.env.action_space.n
    action_sapce = np.array(range(action_sapce_size))
    observation_sapce_size = env_arg.env.observation_space._shape[0]

    # create data space 
    x_input_rtn = np.zeros((num_of_frame_arg, observation_sapce_size+1))
    y_return = np.zeros((num_of_frame_arg, 1))
    y_output_rtn = np.zeros((num_of_frame_arg, 1))

    #define temporal parameters
    reward_sum = 0
    frame = 0
    frame_set = 0
    
    #define hyper parameters
    gamma = 0.95#Q value discount
    alpha = 0.8#Q_function update ratio
    
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
        # action_value 중 최대 값을 선택하면 Q-learnig, action_value의 평균값을 사용하면 mean SARSA

        if mode == 0:
            action_value = action_value_all.mean()#mean SARSA mode
        else : 
            action_value = action_value_all[action_index]# Q_learning mode

        x_input_rtn[i, 0:observation_sapce_size] = observation[:] # 파이썬 인덱싱에 주의할 것. 인덱스가 1:4 이면 끝나는 인덱스가 0번 부터 3번까지
        x_input_rtn[i, observation_sapce_size:observation_sapce_size+1] = action_pick
        if terminated or truncated:
            if terminated == 1:
                reward = -5

            observation, info = env_arg.reset()
            frame_set = i+1
            frame = 0
            reward_sum = 0
        
        y_return[i, :] = reward

        if i % 10 == 0:
            print("{}, {} and {} %".format(i, num_of_frame_arg, round((i/num_of_frame_arg)*100)))
        
        if terminated or truncated == 1:
            break

    for i in range(frame_set):
        return_sum = 0
        for j in range(i, frame_set):
            return_sum +=  y_return[j, :]*(gamma**(j-i))
        print(return_sum)
        
        Q_value = model_arg.predict(x_input_rtn[i].reshape(1,observation_sapce_size+1))
        y_output_rtn[i, :] = (alpha)*Q_value + (1-alpha)*return_sum

    #rolling_queue

    return x_input_rtn, y_output_rtn



def drive_env_by_model(env_arg, num_of_frame_arg, model_arg):    
    observation, info = env_arg.reset()

    # check action and observation space size
    action_sapce_size = env_arg.env.action_space.n
    action_sapce = np.array(range(action_sapce_size))
    observation_sapce_size = env_arg.env.observation_space._shape[0]

    # create data space 
    x_input_rtn = np.zeros((num_of_frame_arg, observation_sapce_size+1))
    y_return = np.zeros((num_of_frame_arg, 1))
    y_output_rtn = np.zeros((num_of_frame_arg, 1))

    #define temporal parameters
    reward_sum = 0
    frame = 0
    frame_set = 0
    
    #define hyper parameters
    gamma = 0.95#Q value discount
    alpha = 0.8#Q_function update ratio
    epsilon = 0
    mode = 0 # 0 : SARSA, 1 : Q_learning

    # pick initial action
    # action_value_all = perdict_model(model_arg, observation, action_sapce)
    # action_index = action_value_all.argmax()
    action_pick = pick_action(model_arg, observation, action_sapce)
    
    for i in range(num_of_frame_arg):
        
        observation, reward, terminated, truncated, info = env_arg.step(action_pick)
        
        #pick next state and action ; which will be used in next iteration 
        # action_value_all = perdict_model(model_arg, observation, action_sapce)
        # action_index = action_value_all.argmax()
        if(np.random.rand() < epsilon):
            action_pick = env_arg.action_space.sample()
        else:
            action_pick = pick_action(model_arg, observation, action_sapce)

        x_input_rtn[i, 0:observation_sapce_size] = observation[:] # 파이썬 인덱싱에 주의할 것. 인덱스가 1:4 이면 끝나는 인덱스가 0번 부터 3번까지
        x_input_rtn[i, observation_sapce_size:observation_sapce_size+1] = action_pick
        if terminated or truncated:
            if terminated == 1:
                reward = -5

            observation, info = env_arg.reset()
            frame_set = i+1
            frame = 0
            reward_sum = 0
        
        y_return[i, :] = reward

        if i % 10 == 0:
            print("{}, {} and {} %".format(i, num_of_frame_arg, round((i/num_of_frame_arg)*100)))
        
        if terminated or truncated == 1:
            break

    for i in range(frame_set):
        return_sum = 0
        for j in range(i, frame_set):
            return_sum +=  y_return[j, :]*(gamma**(j-i))
        print(return_sum)
        print(type(x_input_rtn[i]))
        print(x_input_rtn[i].shape)
        print(x_input_rtn[i])
        Q_value = model_arg.predict(x_input_rtn[i].reshape(1,observation_sapce_size+1))
        y_output_rtn[i, :] = (alpha)*Q_value + (1-alpha)*return_sum
    return x_input_rtn, y_output_rtn



def create_model(env_arg):
    observation_sapce_size = env_arg.env.observation_space._shape[0]

    model_rtn = Sequential()
    model_rtn.add(keras.Input(shape=(observation_sapce_size+1)))
    model_rtn.add(Dense(64, activation='sigmoid'))
    model_rtn.add(Dense(64, activation='relu'))
    model_rtn.add(Dense(16, activation='relu'))
    model_rtn.add(Dense(1, activation='relu'))
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
    model_rtn.add(Dense(128, activation='relu'))
    model_rtn.add(Dense(64, activation='relu'))
    model_rtn.add(Dense(16, activation='relu'))
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
    
    for i in range(action_space_size):#action_space_arg의 열길이에 따라 평가를 반복
        action = action_sapce_arg[i]
        model_input[0, 0:observation_space_size] = observation_arg
        model_input[0, observation_space_size:observation_space_size+1] = action
        
        action_value_rtn[i] = model_arg.predict(model_input, verbose = 0)

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
    env_screen = gym.make("CartPole-v1", render_mode="human")
    env =  gym.make ("CartPole-v1")

    num_of_frame = 100
    # model_glb = create_model(env)
    model_glb = create_model_relu(env)
    # model_glb = create_model_sigmoid(env)


    num_of_frame = 500
    # model_glb = keras.models.load_model("./model/CartPole-v1-uniQ/cart_pole_model_110.h5")
    # model_glb = keras.models.load_model("./model/CartPole-v1-uniQ/cart_pole_model_160.h5")
    model_glb = keras.models.load_model("./model/CartPole-v1-uniQ/cart_pole_model_180.h5")

    for i in range(100):
        x_input, y_output = drive_env_by_model(env_screen, num_of_frame, model_glb)
        # learning_model_batch_10(model_glb, x_input, y_output)
    env_screen.close()
    env.close()