"""
https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=msnayana&logNo=221431225117
"""

# 이 예제... 절차는 정상적이되, 자료를 다루는 구조는 뭐하나 정상적인 것이 벼로 없다. 
# 난잡한 자료를 다루는 중첩적이고 복잡한 방식에 따른 세세한 오류가 너무 많음 
# 이 예제는 버린다. 

import gymnasium as gym
import numpy as np
import random

from keras.models       import Sequential
from keras.layers       import Dense
from keras.optimizers   import Adam

env =  gym.make ("CartPole-v1")
limit_step_glb = 500

sum_score = [] # 뭘 위한 배열?

#Learning function
def study_model(model, data_set_input, data_set_output):
    #지금의 cart_pole 문제는 이미지로 부터 학습하는 것이 아닌 필요한 모든 정보를 숫자로 전달 받으므로, 완전 마크로브한 상태이다. 
    #따라서 별도의 리플레이 버퍼는 필요치 않다. 
    
    print("==============================")
    # print(type(data_set))
    # print(data_set[1][0])
    # print(data_set[1][1])

    
    # print(type(data_set[1][0]))
    
    # print(data_set[1][0].shape)

    # print(type(np.array([i[0] for i in data_set])))
    # print(np.array([i[0] for i in data_set]).shape)
    # print(np.array([i[0] for i in data_set])[1])
    # print(np.array([i[0] for i in data_set]).shape)

    # print(type([i[0] for i in data_set]))
    # print([i[0] for i in data_set].shape)
    
    
    # print(type(np.concatenate([i[0] for i in data_set])))
    # print(np.concatenate([i[0] for i in data_set]).shape)
    # print(np.concatenate([i[0] for i in data_set])[1])

    x_input = data_set_input # 길이에 무관하게 폭을 4개로 만드는 역할
    y_answer = data_set_output

    print("========")
    print(type(x_input))
    print(x_input.shape)
    print(x_input[1])
    print(type(x_input[1]))
    print(type(x_input[1][1]))

    print(type(x_input[0, 0]))
    print(type(x_input[1, 1]))

    print("========")
    # [print(i.shape, i.dtype) for i in x_input]
    # [print(i.shape, i.dtype) for i in x_input[0]]
    # x_input = np.asarray(x_input).astype("float32")
    # print(type(x_input))
    # print(x_input.shape)
    # print(x_input[1])
    # print(type(x_input[1]))
    # print(type(x_input[1][1]))

    

    print("========")
    print(type(y_answer))
    print(y_answer.shape)
    print(y_answer[1])
    print(type(y_answer[1]))
    

    model.fit(x_input, y_answer, epochs = 2)

def get_predict(model, obs):
    return np.random.choice([0, 1], p = model.predict(obs.reshape(-1, 4))[0])


def agent_driving(model, limit_step, i_episode, top_cut, function = lambda a_1, a_2: random.randrange(0, 2), render=False, test=False):
    work_data = np.empty((1, 4)) 
    for i in range (i_episode):
        reward_sum = 0
        work_step = []
        obs = env.reset()
        for step in range (limit_step):
            if render : env.render()
            action = function(model, obs)

            work_step.append((obs, action))
            obs, reward, terminated, truncated, info = env.step(action)
            reward_sum += reward
            if not test:
                if terminated:
                    break

        print("======")
        print(type(obs))
        print(type(action))

        print("======")
        print(type(reward_sum))
        print(type(work_step))

        work_data.append(reward_sum, work_step)
        print(work_data)

    work_data.sort(key=lambda s:-s[0])

    data_set_input = np.empty((1, 4))
    data_set_output = np.empty((1, 2))
    print(data_set_input.shape)
    print(data_set_output.shape)
    dsum = 0
    for i in range(top_cut):
        for step in work_data[i][1]:
            print("======")
            print(np.asarray(step[0], dtype = "float"))
            
            print("======")
            print(np.asarray(step[0], dtype = "float").shape)
            if step[1] == 0:  
                data_set_input = np.append(data_set_input, np.asarray([step[0]]), axis=0)
                data_set_output = np.append(data_set_output, np.asarray([[1, 0]]), axis=0)
            else:             
                data_set_input = np.append(data_set_input, np.asarray([step[0]]), axis=0)
                data_set_output = np.append(data_set_output, np.asarray([[0, 1]]), axis=0)
        dsum += work_data[i][0]
    sum_score.append(dsum/top_cut) 

    
    print(data_set_output.shape)
    return data_set_input, data_set_output
    

if __name__ == '__main__':

    model = Sequential()
    model.add(Dense(128, input_dim=4, activation='relu'))
    model.add(Dense(52, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='mse', optimizer=Adam())

    data_set_input_rtn, data_set_output_rtn = agent_driving(model, limit_step_glb, 60, 10)
    study_model(model, data_set_input_rtn, data_set_output_rtn)

    for i in range(6):
        data_set_input_rtn, data_set_output_rtn = agent_driving(model, limit_step_glb, 60, 10, get_predict)    
        study_model(model, data_set_input_rtn, data_set_output_rtn)

    print("avg : %s" % (sum_score))
    model.save('test.h5')

    #model=load_model('test.h5')    
    agent_driving(model, limit_step_glb, 100, 60, get_predict, True, True)
    env.close()