import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

import time


from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

"""
cart pole 문제를 deepmind 식 DQN으로 해결하기

특징 1
    DQN
특징 2
    action 출력이 여러개인 네트워크
특징 3
    replay buffer

    #특히 lunarlander 문제에서는 시간에 대한 연관성이 커지며 중요성이 높아질 듯. 

기타:
    가능하면 lunarlander 와 호환 가능하도록
    
"""

class DQN_MeanSquaredError_custom(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        loss = tf.reduce_mean((y_true-y_pred)**2)
        return loss 

class dqn_agent():
    model = 0
    env = 0

    #model spec value
    # action_model = 0
    # target_model = 0

    replaymemory = 0

    #env spec value
    observation_sapce_size = 0
    action_space_size = 0
    action_space = 0

    

    def __init__(self):
        self.reset_minibathch()

        return
    
    def drive_random(self):
        return
    
    
    def drive_model(self):
        print("drive_model")
        # 각 단계에서의 행동선택
        # ★하나의 액션에 대하여 한정하여  loss를 계산하는 방법은
            # 선택 된 액션 이외의 교사값은 모델의 예측갑을 그대로 전달하여 같게하고, 
            # 선택된 액션의 교사값만은 실제값을 전달해 예측값과 실제값 차이를 계산하여 오차로 한다.
            # 이를 통해 Loss 계산에서 다른 액션의 항에 의한 영향들은 모두 0이 되므로 Loss 에는 한 행동의 영향만이 적용된다.   
        # target 생성

        #상수 선언
        # gamma = self.gamma
        # epsilon = self.epsilon
        gamma = 0.99
        epsilon = 0.1
        C_step = 20
        # 초기 상태
        # 필요 요소, S 초기화
        self.reset_minibathch()#replay buffer 초기화
        observation_0_sequence = np.empty([self.observation_sapce_size, self.sequence_length])
        observation_1_sequence = np.empty([self.observation_sapce_size, self.sequence_length])
        observation_1, info = self.env.reset()
        # print(action_0)
        for i in range(19):#시나리오의 최대 길이 500/1000 만큼 반복
        # 반복 상태 
        # 필요 요소, RSA
            #state_0  = state_1 다음 상황으로 넘어감.
            observation_0 = observation_1
            observation_0_input = observation_0.reshape([1, self.observation_sapce_size])
            Q_value_0 =  self.action_model.predict(observation_0_input, verbose = 0)#Q_value from behaivior policy
            # print(Q_value_0)
            action = self.pick_action(Q_value_0, epsilon=0.1)

            observation_1, reward, terminated, tuncated, info = self.env.step(action)
            #replay buffer
            # 저장 요소 : SARS 4가질로 충분, 이유는 a_t1은 s_t1 으로 부터 유도 가능.
            observation_0_sequence[:, 0] = observation_0
            observation_1_sequence[:, 0] = observation_1

            self.push_minibatch(observation_0_sequence, reward, action, observation_1_sequence)
            
            self.get_minibatch_mass()
            # if i % C_step == 4:
                # train_set = self.get_train_set(5)
                # self.pop_minibatch()               
                # self.pop_minibatch()               
                # self.pop_minibatch()               
                # self.pop_minibatch()               
                # self.pop_minibatch()               

        # observation, reward, terminated, turncated, info = self.env.step(0)
        print("----------")
        self.get_minibatch_mass()
        print(self.get_minibatch(0))
        print("----------")
        self.get_minibatch_mass()
        print(self.get_minibatch(0))
        print("----------")
        self.get_minibatch_mass()
        print(self.get_minibatch(5))
        print("----------")
        self.get_minibatch_mass()
        print(self.get_minibatch(19))
        print("----------")
        self.get_minibatch_mass()
        return Q_value_0
    
    # def get_model(self, model_arg): # model 의 생성과 관리는 클래스 내부에서 처리.
    #     return

    def pick_action(self, Q_value, epsilon):
        """
        input : Q_value : type np.array(1, None) 2 demension
        """
        self.action_space
        
        if np.random.rand() > epsilon:
            action_index = Q_value.argmax()# 타입 
        else:
            action_index = np.random.choice(self.action_space)
        action  = self.action_space[action_index]
        return action
    
    def reset_minibathch(self):#minibatch
        self.sequence_length = 1
        self.queue_length = 20
        self.queue_front = 0
        self.queue_rear = 0
        self.queue_full_tag = 0#0 : not full, 1 : full
        
        queue_length = self.queue_length
        self.seqeunce_0 = np.empty([self.queue_length, self.observation_sapce_size, self.sequence_length])
        self.action_0 = np.empty([self.queue_length, 1])
        self.reward_0 = np.empty([self.queue_length, 1])
        self.seqeunce_1 = np.empty([queue_length, self.observation_sapce_size, self.sequence_length])

    def push_minibatch(self, seqeunce_0_arg, action_0_arg, reward_0_arg, seqeunce_1_arg):

        # index = self.batch_counter%self.queue_length
        if self.queue_full_tag == 0:
            self.queue_rear = (self.queue_rear+1)%self.queue_length
            index = self.queue_rear
            if self.queue_front == self.queue_rear:
                self.queue_full_tag = 1

        else:
            #이미 큐가 꽉찬 상태.
            #큐를 이동하며 기존의 내용을 덮어쓴다.
            self.queue_rear = (self.queue_rear+1)%self.queue_length
            self.queue_front = self.queue_rear#
            index = self.queue_rear
            self.queue_full_tag = 2

        self.seqeunce_0[index, :, :] = seqeunce_0_arg[:, :]
        self.action_0[index, :] = action_0_arg
        self.reward_0[index, :] = reward_0_arg
        self.seqeunce_1[index, :, :] = seqeunce_1_arg[:, :]

        return self.queue_full_tag
    
    def pop_minibatch(self):
        seqeunce_0_rtn = 0
        action_0_rtn = 0
        reward_0_rtn = 0
        seqeunce_1_rtn = 0
        if self.queue_front == self.queue_rear:
            if self.queue_full_tag == 0:
                #이미 큐가 텅 빈 상태.
                #있는 값은 덮어 써도 없는 값은 꺼낼 수 없다.
                raise Exception("큐 비었음")#큐 비었음.
        else:
            self.queue_front = (self.queue_front+1)%self.queue_length
            index = self.queue_front
            if self.queue_front == self.queue_rear:
                self.queue_full_tag = 0

            seqeunce_0_rtn = self.seqeunce_0[index, :, :]
            action_0_rtn = self.action_0[index, :]
            reward_0_rtn = self.reward_0[index, :]
            seqeunce_1_rtn = self.seqeunce_1[index, :, :]
            # print("{} : {}".format(self.queue_front, self.queue_rear))

            #delete element by overiding empty space
            self.seqeunce_0[index, :, :] = np.empty([self.observation_sapce_size, self.sequence_length])
            self.action_0[index, :] = np.empty([1, 1])
            self.reward_0[index, :] = np.empty([1, 1])
            self.seqeunce_1[index, :, :] = np.empty([self.observation_sapce_size, self.sequence_length])

        return seqeunce_0_rtn, action_0_rtn, reward_0_rtn, seqeunce_1_rtn

    def get_minibatch_mass(self):#num of stored data
        if self.queue_full_tag > 0:
            num_of_element = self.queue_length# 꽉찬 상태
        else:
            if self.queue_front == self.queue_rear:#텅빈상태
                num_of_element = 0
            else:
                #개수를 직접 새어야 하는 꽉차지도 텅비지고 않은 상태
                # if self.queue_rear > self.queue_front:
                #     (self.queue_length + self.queue_rear - self.queue_front)%self.queue_length
                # else:
                #     self.queue_rear - self.queue_front

                num_of_element = (self.queue_length + self.queue_rear - self.queue_front)%self.queue_length# 개수 새기
        
        print("{} : {} : {}".format(num_of_element, self.queue_front, self.queue_rear))
        return num_of_element


    def set_minibatch(self, index, seqeunce_0_arg, action_0_arg, reward_0_arg, seqeunce_1_arg):
        # index = self.batch_counter%self.queue_length
                # index = self.batch_counter%self.queue_length
        num_of_elements = self.get_minibatch_mass()

        if index < num_of_elements:
            index_circle = (self.queue_front + index)%self.queue_length
        
            self.seqeunce_0[index, :, :] = seqeunce_0_arg[:, :]
            self.action_0[index, :] = action_0_arg[:, :]
            self.reward_0[index, :] = reward_0_arg[:, :]
            self.seqeunce_1[index, :, :] = seqeunce_1_arg[:, :]
        
        else:
            raise Exception("out of index")


    def get_minibatch(self, index):           
        num_of_elements = self.get_minibatch_mass()

        # index = self.batch_counter%self.queue_length
        if index < num_of_elements:
            index_circle = (self.queue_front + index)%self.queue_length
        
            seqeunce_0_rtn = self.seqeunce_0[index_circle, :, :]
            action_0_rtn = self.action_0[index_circle, :]
            reward_0_rtn = self.reward_0[index_circle, :]
            seqeunce_1_rtn = self.seqeunce_1[index_circle, :, :]
        else:
            raise Exception("out of index")
        return seqeunce_0_rtn, action_0_rtn, reward_0_rtn, seqeunce_1_rtn

    def get_train_set(self, batch_size):
        seqeunce_0_batch_rtn = np.empty(batch_size, self.sequence_length)
        action_0_batch_rtn = np.empty(batch_size, 1)
        reward_0_batch_rtn = np.empty(batch_size, 1)
        seqeunce_1_batch_rtn = np.empty(batch_size, self.sequence_length)

        queue_size = self.get_minibatch_mass()

        index = np.random.choice(range(queue_size), batch_size)
        index_list = index.tolist()
        i = 0
        for index in index_list:
            seqeunce_0_rtn, action_0_rtn, reward_0_rtn, seqeunce_1_rtn = self.get_minibatch(index)
            seqeunce_0_batch_rtn[i, :] = seqeunce_0_rtn[0, :]
            action_0_batch_rtn[i, :] = action_0_rtn[0, :]
            reward_0_batch_rtn[i, :] = reward_0_rtn[1, :]
            seqeunce_1_batch_rtn[i, :] = seqeunce_1_rtn[1, :]
            i+1

        return seqeunce_0_batch_rtn, action_0_batch_rtn, reward_0_batch_rtn, seqeunce_1_batch_rtn
    
    def mask_target(self):
        
        return
    
    
    def set_env(self, env_arg : gym.Env):
        self.env = env_arg
        self.inspect_env()
    
    def inspect_env(self):
        self.observation_sapce_size = self.env.observation_space._shape[0]
        self.action_space_size = self.env.action_space.n
        self.action_space = np.array(range(self.action_space_size))

    def create_nn(self):
        #custom loss function

        self.action_model = tf.keras.Sequential()
        self.action_model.add(tf.keras.Input(shape=(self.observation_sapce_size)))#입력 레이어
        self.action_model.add(tf.keras.layers.Dense(128, activation='relu'))
        self.action_model.add(tf.keras.layers.Dense(64, activation='relu'))
        self.action_model.add(tf.keras.layers.Dense(16, activation='relu'))
        self.action_model.add(tf.keras.layers.Dense(self.action_space_size, activation='relu'))#출력 레이어
        self.action_model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD())

        self.target_model = tf.keras.Sequential()
        self.target_model.add(tf.keras.Input(shape=(self.observation_sapce_size)))#입력 레이어
        self.target_model.add(tf.keras.layers.Dense(128, activation='relu'))
        self.target_model.add(tf.keras.layers.Dense(64, activation='relu'))
        self.target_model.add(tf.keras.layers.Dense(16, activation='relu'))
        self.target_model.add(tf.keras.layers.Dense(self.action_space_size, activation='relu'))#출력 레이어
        self.target_model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD())

        self.target_model.summary()

        self.action_model.set_weights(self.target_model.get_weights())

    def weights_copy(self):
        self.action_model.set_weights(self.target_model.get_weights())


if __name__ == "__main__":
    # env_headless = gym.make("CartPole-v1", render_mode="human")
    # env_screen = gym.make("CartPole-v1")
    env_headless = gym.make("LunarLander-v2", render_mode="human")
    env_screen = gym.make("LunarLander-v2")

    agent = dqn_agent()

    agent.set_env(env_screen)
    agent.create_nn()
    agent.drive_model()

    env_headless.reset()
    time.sleep(3)
    env_headless.close()
    env_screen.close()

