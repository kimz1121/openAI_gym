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

    #train hyperparameter
    gamma = 0.99
    epsilon = 0.1
    epsilon_decay = 1
    epsilon_min = 0.01
    alpha = 0

    C_step_counter = 0
    C_step = 5
    tau = 0.0001

    batch_size = 5
    sequence_length = 1
    queue_length = 10000

    def __init__(self, env_arg):
        C_step_counter = 0
        self.set_env(env_arg)
        self.inspect_env()
        self.reset_minibathch()
    
    def set_hyper_parameter(self, gamma, epsilon, alpha, tau, C_step):
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.tau = tau
        self.C_step = C_step

    def set_epslion_decay(self, epsilon_decay, epsilon_min):
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        ...

    def do_epsilon_decay(self):
        if self.epsilon_decay < 1:
            if self.epsilon > self.epsilon_min:
                self.epsilon = self.epsilon*self.epsilon_decay
            if self.epsilon <= self.epsilon_min:
                self.epsilon = self.epsilon_min
    
    def set_tau_decay(self, epsilon_decay, epsilon_min):
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        ...

    def do_tau_decay(self):
        if self.epsilon_decay < 1:
            if self.epsilon > self.epsilon_min:
                self.epsilon = self.epsilon*self.epsilon_decay
            if self.epsilon <= self.epsilon_min:
                self.epsilon = self.epsilon_min

    def set_tau_Cstep_temp(self, tau, C_step):
        self.tau = tau
        self.C_step = C_step
    
    def drive_queue_init(self):
        #replay 버퍼 초기화를 위한 임의동작   
        print("버퍼 초기화 시작")     
        gamma = self.gamma
        epsilon = self.epsilon
        C_step = self.C_step
        observation_0_sequence = np.empty([self.observation_sapce_size, self.sequence_length])
        observation_1_sequence = np.empty([self.observation_sapce_size, self.sequence_length])
        observation_1, info = self.env.reset()
        iter = self.queue_length
        for i in range(iter):
            print("버퍼 초기화 {:3}% 완료".format(round(100*(i/iter))))
            observation_0 = observation_1
            observation_0_input = observation_0.reshape([1, self.observation_sapce_size])
            Q_value_0 =  self.action_model.predict(observation_0_input, verbose = 0)
            action = self.pick_action(Q_value_0, epsilon=1)#완전한 무작위
            observation_1, reward, terminated, truncated, info = self.env.step(action)
            observation_0_sequence[:, 0] = observation_0
            observation_1_sequence[:, 0] = observation_1

            self.push_minibatch(observation_0_sequence, action, reward, observation_1_sequence, terminated)
            
            # 단순 버퍼 초기화가 목적이므로 학습은 진행하지 않음
            # sample_set = self.get_minibatch_random_sample(self.batch_size)
            # x_input, y_output = self.get_train_set(self.batch_size, *sample_set)
            # self.action_model.fit(x_input, y_output, batch_size=self.batch_size, epochs = 2)
            # self.get_minibatch_mass()
            # if i % C_step == 0:#for each C_step
                # self.weights_copy()

            # 버퍼 초기화를 위해 종료가 아닌 env.reset()을 수행
            if terminated or truncated == 1:
                self.env.reset()
        print("버퍼 초기화 완료")

    def drive_model(self):
        # 각 단계에서의 행동선택
        # ★하나의 액션에 대하여 한정하여  loss를 계산하는 방법은
            # 선택 된 액션 이외의 교사값은 모델의 예측갑을 그대로 전달하여 같게하고, 
            # 선택된 액션의 교사값만은 실제값을 전달해 예측값과 실제값 차이를 계산하여 오차로 한다.
            # 이를 통해 Loss 계산에서 다른 액션의 항에 의한 영향들은 모두 0이 되므로 Loss 에는 한 행동의 영향만이 적용된다.   
        # target 생성

        #상수 선언
        gamma = self.gamma
        epsilon = self.epsilon
        C_step = self.C_step
        # gamma = 0.99
        # epsilon = 0.1
        # C_step = 20
        # 초기 상태
        # 필요 요소, S 초기화
        # self.reset_minibathch()#replay buffer 초기화
        observation_0_sequence = np.empty([self.observation_sapce_size, self.sequence_length])
        observation_1_sequence = np.empty([self.observation_sapce_size, self.sequence_length])
        observation_1, info = self.env.reset()
        # print(action_0)
        reward_sum = 0
        for i in range(1000):#시나리오의 최대 길이 500/1000 만큼 반복
        # 반복 상태 
        # 필요 요소, RSA
            #state_0  = state_1 다음 상황으로 넘어감.
            observation_0 = observation_1
            observation_0_input = observation_0.reshape([1, self.observation_sapce_size])
            Q_value_0 =  self.action_model.predict(observation_0_input, verbose = 0)#Q_value from behaivior policy
            # print(Q_value_0)
            action = self.pick_action(Q_value_0, epsilon=0.1)
            observation_1, reward, terminated, truncated, info = self.env.step(action)
            #calc reward_sum
            reward_sum += reward
            #storing to replay buffer
            # 저장 요소 : SARS 4가질로 충분, 이유는 a_t1은 s_t1 으로 부터 유도 가능.
            observation_0_sequence[:, 0] = observation_0
            observation_1_sequence[:, 0] = observation_1

            self.push_minibatch(observation_0_sequence, action, reward, observation_1_sequence, terminated)
            #sampling from replay buffer
            sample_set = self.get_minibatch_random_sample(self.batch_size)
            x_input, y_output = self.get_train_set(self.batch_size, *sample_set)#* 언패킹 대상은 s_0, a_0, r_0 s_1 이다.
        
            self.action_model.fit(x_input, y_output, batch_size=10, epochs = 1, verbose=0)
            
            self.get_minibatch_mass()
            
            self.C_step_counter += 1
            if self.C_step_counter % C_step == 0:#for each C_step
                # print("copy!!")
                self.C_step_counter = 0
                self.weights_copy()

            if terminated or truncated == 1:
                # print("reward_total : {}".format(reward_sum))
                break

            self.do_epsilon_decay()

        return reward_sum
    # def get_model(self, model_arg): # model 의 생성과 관리는 클래스 내부에서 처리.
    #     return

    def drive_model_saved(self):        
        # 학습 없이 구동만.
        observation_0_sequence = np.empty([self.observation_sapce_size, self.sequence_length])
        observation_1_sequence = np.empty([self.observation_sapce_size, self.sequence_length])
        observation_1, info = self.env.reset()
        # print(action_0)
        reward_sum = 0
        for i in range(1000):#시나리오의 최대 길이 500/1000 만큼 반복
        # 반복 상태 
        # 필요 요소, RSA
            #state_0  = state_1 다음 상황으로 넘어감.
            observation_0 = observation_1
            observation_0_input = observation_0.reshape([1, self.observation_sapce_size])
            Q_value_0 =  self.action_model.predict(observation_0_input, verbose = 0)#Q_value from behaivior policy
            # print(Q_value_0)
            action = self.pick_action(Q_value_0, epsilon=0.1)
            observation_1, reward, terminated, truncated, info = self.env.step(action)
            #calc reward_sum
            reward_sum += reward
            #storing to replay buffer
            # 저장 요소 : SARS 4가질로 충분, 이유는 a_t1은 s_t1 으로 부터 유도 가능.
            observation_0_sequence[:, 0] = observation_0
            observation_1_sequence[:, 0] = observation_1
            
            if terminated or truncated == 1:
                # print("reward_total : {}".format(reward_sum))
                break

        return reward_sum

    def pick_action(self, Q_value, epsilon):
        """
        input : Q_value : type np.array(1, None) 2 demension
        """
        if np.random.rand() > epsilon:
            action_index = Q_value.argmax()# 타입
        else:
            action_index = np.random.choice(self.action_space)
        action  = self.action_space[action_index]
        return action
    
    def reset_minibathch(self):#minibatch
        self.queue_front = 0
        self.queue_rear = 0
        self.queue_full_tag = 0#0 : not full, 1 : full

        queue_length = self.queue_length
        self.seqeunce_0 = np.empty([self.queue_length, self.observation_sapce_size, self.sequence_length], dtype = "float64")
        self.action_0 = np.empty([self.queue_length, 1], dtype = "int64")
        self.reward_0 = np.empty([self.queue_length, 1], dtype = "float64")
        self.seqeunce_1 = np.empty([queue_length, self.observation_sapce_size, self.sequence_length], dtype = "float64")
        self.terminated = np.empty([self.queue_length, 1], dtype = "bool")

    def push_minibatch(self, seqeunce_0_arg, action_0_arg, reward_0_arg, seqeunce_1_arg, terminated_arg):

        # index = self.batch_counter%self.queue_length
        index = self.queue_rear
        if self.queue_full_tag == 0:
            self.queue_rear = (self.queue_rear+1)%self.queue_length
            if self.queue_front == self.queue_rear:
                self.queue_full_tag = 1
        else:
            # 이미 큐가 꽉찬 상태.
            # self.queue_front == self.queue_rear 
            # index 값을 front에서 가져오나 rear에서 가져오나 결과는 같다.
            # 큐를 이동하며 기존의 내용을 덮어쓴다.
            self.queue_rear = (self.queue_rear+1)%self.queue_length
            self.queue_front = self.queue_rear#
            self.queue_full_tag = 2

        self.seqeunce_0[index, :, :] = seqeunce_0_arg[:, :]
        self.action_0[index, :] = action_0_arg
        self.reward_0[index, :] = reward_0_arg
        self.seqeunce_1[index, :, :] = seqeunce_1_arg[:, :]
        self.terminated[index, :] = terminated_arg

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
            index = self.queue_front
            self.queue_front = (self.queue_front+1)%self.queue_length
            if self.queue_front == self.queue_rear:
                self.queue_full_tag = 0

            seqeunce_0_rtn = self.seqeunce_0[index, :, :]
            action_0_rtn = self.action_0[index, :]
            reward_0_rtn = self.reward_0[index, :]
            seqeunce_1_rtn = self.seqeunce_1[index, :, :]
            terminated_rtn = self.terminated[index, :]
            # print("{} : {}".format(self.queue_front, self.queue_rear))

            #delete element by overiding empty space
            self.seqeunce_0[index, :, :] = np.empty([self.observation_sapce_size, self.sequence_length], dtype = "float64")
            self.action_0[index, :] = np.empty([1, 1], dtype = "int64")
            self.reward_0[index, :] = np.empty([1, 1], dtype = "float64")
            self.seqeunce_1[index, :, :] = np.empty([self.observation_sapce_size, self.sequence_length], dtype = "float64")
            self.terminated[index, :] = np.empty([1, 1], dtype = "bool")

        return seqeunce_0_rtn, action_0_rtn, reward_0_rtn, seqeunce_1_rtn, terminated_rtn

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
        
        # print("{} : {} : {}".format(num_of_element, self.queue_front, self.queue_rear))
        return num_of_element


    def set_minibatch(self, index, seqeunce_0_arg, action_0_arg, reward_0_arg, seqeunce_1_arg, terminated_arg):
        # index = self.batch_counter%self.queue_length
                # index = self.batch_counter%self.queue_length
        num_of_elements = self.get_minibatch_mass()

        if index < num_of_elements:
            index_circle = (self.queue_front + index)%self.queue_length
        
            self.seqeunce_0[index, :, :] = seqeunce_0_arg[:, :]
            self.action_0[index, :] = action_0_arg
            self.reward_0[index, :] = reward_0_arg
            self.seqeunce_1[index, :, :] = seqeunce_1_arg[:, :]
            self.terminated[index, :] = terminated_arg
        
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
            terminated_rtn = self.terminated[index_circle, :]
        else:
            raise Exception("out of index")
        
        
        return seqeunce_0_rtn, action_0_rtn, reward_0_rtn, seqeunce_1_rtn, terminated_rtn

    def get_minibatch_random_sample(self, batch_size):
        seqeunce_0_batch_rtn = np.empty([batch_size, self.observation_sapce_size, self.sequence_length], dtype = "float64")
        action_0_batch_rtn = np.empty([batch_size, 1], dtype = "int64")
        reward_0_batch_rtn = np.empty([batch_size, 1], dtype = "float64")
        seqeunce_1_batch_rtn = np.empty([batch_size, self.observation_sapce_size, self.sequence_length], dtype = "float64")
        terminated_batch_rtn = np.empty([batch_size, 1], dtype = "bool")

        queue_size = self.get_minibatch_mass()

        index = np.random.choice(range(queue_size), batch_size)
        index_list = index.tolist()
        for i in range(batch_size):
            seqeunce_0_rtn, action_0_rtn, reward_0_rtn, seqeunce_1_rtn, terminated_rtn = self.get_minibatch(index_list[i])

            seqeunce_0_batch_rtn[i, :, :] = seqeunce_0_rtn[:, :]
            action_0_batch_rtn[i, :] = action_0_rtn
            reward_0_batch_rtn[i, :] = reward_0_rtn
            seqeunce_1_batch_rtn[i, :, :] = seqeunce_1_rtn[:, :]
            terminated_batch_rtn[i, :] = terminated_rtn

        return seqeunce_0_batch_rtn, action_0_batch_rtn, reward_0_batch_rtn, seqeunce_1_batch_rtn, terminated_batch_rtn
    
    def get_train_set(self, batch_size, seqeunce_0_batch_arg, action_0_batch_arg, reward_0_batch_arg, seqeunce_1_batch_arg, terminated_batch_arg):
        #s_0
        input_space_action = np.empty([batch_size, self.observation_sapce_size*self.sequence_length])
        #s_1
        input_space_target = np.empty([batch_size, self.observation_sapce_size*self.sequence_length])
        #Q_value true
        output_space = np.empty([batch_size, self.action_space_size])

        for i in range(batch_size):# faltten obserbation
            # for each action
            for j in range(self.sequence_length):
                #for sequence
                #s_0
                input_space_action[i, j*self.observation_sapce_size:(j+1)*self.observation_sapce_size] = seqeunce_0_batch_arg[i, :, j]
                #s_1
                input_space_target[i, j*self.observation_sapce_size:(j+1)*self.observation_sapce_size] = seqeunce_1_batch_arg[i, :, j]
            
            #y_j = G_j
            if terminated_batch_arg[i, 0] == 0:
            # = r_j + G_j+1
            # = r_j + Q(s_1, a_1 theta-) : theta- weight of target network
            
                Q_value_1 =  self.target_model.predict(input_space_target[i, :].reshape([1, self.observation_sapce_size]), verbose = 0)
                action_1 = Q_value_1.argmax()#greedy-action

                y_j = reward_0_batch_arg[i] + self.gamma*Q_value_1[0, action_1]#add reward for the action_0
            else:
                y_j = reward_0_batch_arg[i]

            #Loss 
            # = y_j - Q(s_0, a_0 theta) : theta- weight of action network
            Q_value_0 =  self.action_model.predict(input_space_action[i, :].reshape([1, self.observation_sapce_size]), verbose = 0)
            # calc loss for only action_0
            # by a method shows Q_value which is more exact
            # 더 정확히 액션 밸류를 평가하기 위해서는, 행동 정책을 통해 reward가 반영된 actoin에 대하여서만 
            # 액션 밸류를 업데이트 하여야, 밸만 방정식을 더욱 엄밀히 충족한다. 
            # S_1 상태에서 얻어진 각 action에 대한 Q_value 들의 reward가 action_0 만 존재하기 때문에 
            # 다른 action 에 대한 Q_value 들은 reward를 합산하지 못해 밸만 방정식에 어긋난다. 
            # update Q_value only about action_0 

            action_0 = action_0_batch_arg[i, 0]#epsilon-greey and the action what the machine choosed in state_0
            #soft update
            Q_value_0[0, action_0] = (self.alpha)*Q_value_0[0, action_0] + (1-self.alpha)*y_j
            output_space[i, :] = Q_value_0[0, :]

        # 반환하는 결과
        # machine이 겪은 상황 : state_0, machine 이 주어진 상황에서 예측해야할 올바른 Q_value_0
        # input_space = state_0
        # output_space = y_j = Q(s_0, a_0 theta-) # TD-target
        # y_rediction_of_machine =  Q(s_0, a_0 theta)

        return input_space_action, output_space

    def mask_target(self, measure_q, predict_q):

        return
    
    
    def set_env(self, env_arg : gym.Env):
        self.env = env_arg
        self.inspect_env()
    
    def inspect_env(self):
        self.observation_sapce_size = self.env.observation_space._shape[0]
        self.action_space_size = self.env.action_space.n
        self.action_space = np.array(range(self.action_space_size), dtype="int")

    def create_nn(self):
        #custom loss function

        self.action_model = tf.keras.Sequential()
        self.action_model.add(tf.keras.Input(shape=(self.observation_sapce_size*self.sequence_length)))#입력 레이어
        # self.action_model.add(tf.keras.layers.Dense(128, activation='relu'))
        self.action_model.add(tf.keras.layers.Dense(64, activation='relu'))
        self.action_model.add(tf.keras.layers.Dense(64, activation='relu'))
        self.action_model.add(tf.keras.layers.Dense(64, activation='relu'))
        # self.action_model.add(tf.keras.layers.Dense(16, activation='relu'))
        self.action_model.add(tf.keras.layers.Dense(self.action_space_size, activation='linear'))#출력 레이어
        # self.action_model.compile(loss='mse', optimizer=tf.keras.optimizers.legacy.SGD())
        # self.action_model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.legacy.SGD())
        self.action_model.compile(loss='mse', optimizer=tf.keras.optimizers.legacy.Adam())
        # self.action_model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.legacy.Adam())

        self.target_model = tf.keras.Sequential()
        self.target_model.add(tf.keras.Input(shape=(self.observation_sapce_size*self.sequence_length)))#입력 레이어
        # self.target_model.add(tf.keras.layers.Dense(128, activation='relu'))
        self.target_model.add(tf.keras.layers.Dense(64, activation='relu'))
        self.target_model.add(tf.keras.layers.Dense(64, activation='relu'))
        self.target_model.add(tf.keras.layers.Dense(64, activation='relu'))
        # self.target_model.add(tf.keras.layers.Dense(16, activation='relu'))
        self.target_model.add(tf.keras.layers.Dense(self.action_space_size, activation='linear'))#출력 레이어
        # self.target_model.compile(loss='mse', optimizer=tf.keras.optimizers.legacy.SGD())
        # self.target_model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.legacy.SGD())
        self.target_model.compile(loss='mse', optimizer=tf.keras.optimizers.legacy.Adam())
        # self.target_model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.legacy.Adam())


        self.target_model.summary()

        self.action_model.set_weights(self.target_model.get_weights())

    def weights_copy(self):
        target_weights = self.target_model.get_weights()
        action_weights = self.action_model.get_weights()

        # print(len(target_weights))
        update_weights = [None]*len(target_weights)
        for i in range(len(target_weights)):
            # print(type(target_weights[i]))
            # print(target_weights[i].shape)

        #     update_weights[i] = self.tau*target_weights[i] + (1 - self.tau)*action_weights[i]
        # self.action_weights.set_weights(update_weights)
            update_weights[i] = self.tau*action_weights[i] + (1 - self.tau)*target_weights[i]
        self.target_model.set_weights(update_weights)# 실수 박제

    def save_model(self, generation, index):
        self.action_model.save("D:/ksw_coding/python/openAI_gym/model/lunarlander-v2/genetation-{}/action_{}.h5".format(generation, index))
        self.target_model.save("D:/ksw_coding/python/openAI_gym/model/lunarlander-v2/genetation-{}/target_{}.h5".format(generation, index))

    def load_model(self, generation, index):
        self.action_model = keras.models.load_model("D:/ksw_coding/python/openAI_gym/model/lunarlander-v2/genetation-{}/action_{}.h5".format(generation, index))
        self.target_model = keras.models.load_model("D:/ksw_coding/python/openAI_gym/model/lunarlander-v2/genetation-{}/action_{}.h5".format(generation, index))

if __name__ == "__main__":
    # env_screen = gym.make("CartPole-v1", render_mode="human")
    # env_headless = gym.make("CartPole-v1")
    env_screen = gym.make("LunarLander-v2", render_mode="human")
    env_headless = gym.make("LunarLander-v2")


    agent = dqn_agent(env_screen)
    agent.set_env(env_headless)
    
    agent.set_hyper_parameter(gamma=0.99, epsilon=0.1, alpha=0, tau=0.001, C_step=1)
    agent.set_epslion_decay(epsilon_decay=0.995, epsilon_min=0.001)
    
    agent.create_nn()
    
    agent.weights_copy()

    # agent.drive_queue_init()

    iter_max = 1000000
    generation = 39


    reward_list = []
    for i in range(iter_max):
        if i%10 == 0:
            agent.set_env(env_screen)
        else:
            agent.set_env(env_headless)

        reward_sum = agent.drive_model()

        if len(reward_list) >= 25:
            reward_list.pop(0)
        reward_list.append(reward_sum)

        reward_mean = sum(reward_list)/len(reward_list)
        print("iter : {:10}/{}  || R_sum {:5.3f} || R_mean {:5.3f}".format(i, iter_max, reward_sum, reward_mean))

        if i%25 == 0:
            agent.save_model(generation, i)

        if reward_mean >= 250:
            agent.save_model(generation, i)
            print("leaning complete")
            break

        if i > 750:
            agent.set_tau_Cstep_temp(tau=0.001, C_step= 5)

        if i > 1000:
            agent.set_tau_Cstep_temp(tau=0.0001, C_step=10)

    # agent.set_env(env_screen)
    # agent.load_model(36, 2600)
    # for i in range(100):
    #     agent.drive_model_saved()

    env_headless.reset()
    time.sleep(3)
    env_headless.close()
    env_screen.close()

