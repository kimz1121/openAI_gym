"""
https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=msnayana&logNo=221431225117
"""

import gym
import numpy as np
import random

env = gym.make('CartPole-v1')
limit_step = 500

sum_score = []

from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam

def study_model(model, data_set):
  X = np.array([i[0] for i in data_set]).reshape(-1, 4)
  y = np.array([i[1] for i in data_set]).reshape(-1, 2)
  model.fit(X, y, epochs=2)

def get_predict(s):
    return np.random.choice([0, 1], p=model.predict(s.reshape(-1, 4))[0] )

#==============================================
def agent_driving(i_episode,top_cut, fun, render=False, test=False):
  work_data = []
  for i in range(i_episode):
    reward_sum = 0
    work_step = []
    obs = env.reset()
    for step in range(limit_step):
      if render: env.render()
      action = fun(obs)
      work_step.append((obs, action))
      obs, reward, terminated, truncated, info = env.step(action)
      reward_sum += reward
      if not test:
        if terminated:
          break
    work_data.append((reward_sum, work_step))
  
  work_data.sort(key=lambda s:-s[0])

  # summary for data_set
  data_set = []
  dsum = 0
  for i in range(top_cut):
    for step in work_data[i][1]:
      if step[1] == 0:  data_set.append((step[0], [1, 0]))
      else:             data_set.append((step[0], [0, 1]))
    dsum += work_data[i][0]
  sum_score.append(dsum/top_cut) 
  return data_set
#===================================


#===================================  
if __name__ == '__main__':

  model = Sequential()
  model.add(Dense(128, input_dim=4, activation='relu'))
  model.add(Dense(52, activation='relu'))
  model.add(Dense(2, activation='softmax'))
  model.compile(loss='mse', optimizer=Adam())

  get_data = agent_driving(60, 10, lambda s: random.randrange(0, 2))
  study_model(model, get_data)

  for i in range(6):#incremental MC 방식으로 학습 : 장점 간단하다. 모델프리 학습가능.
    get_data = agent_driving(60, 10, get_predict)    
    study_model(model, get_data)

  print("avg : %s" % (sum_score))
  model.save('test.h5')

  #model=load_model('test.h5')    
  agent_driving(100, 60, get_predict, True, True)
  env.close()
#===================================