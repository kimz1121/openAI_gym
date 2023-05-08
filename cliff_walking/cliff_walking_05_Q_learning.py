import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


class Agent:
    Q_table = np.zeros((12, 4, 4))
    # observation = None
    # reward = None
    # terminated = None
    # turncated = None
    # info = None

    def __init__(self):
        print(type(self.Q_table))
        print(self.Q_table.shape)

    def put_gym_env(self, env_arg: gym.Env):
        self.env = env_arg


    def drive_sarsa_random(self):
        
        return
    
    def drive_sarsa_mean_tabel(self, epsilon_arg):
        ation_value_temp = np.empty(4)
        observation_t0, info = self.env.reset()

        action = self.pick_action(observation_t0, epsilon=epsilon_arg)

        
        # sarsa hyper parameter
        gamma = 0.9
        alpha = 0.8

        iter_limit = 100000
        i = 0
        while 1:
            Q_value_0 = self.get_Q_value(observation_t0, action)
            observation_t1, reward, terminated, turncated, info = self.env.step(action)

            Q_value_1 = self.get_Q_value_all(observation_t1)
            Q_value_1_mean = float(Q_value_1.mean())

            # clac Q_value target
            if terminated or turncated:
                if terminated:
                    reward = 1000
                print("===========================")
                print(reward)
                Q_value_bellman = reward
            else:
                Q_value_bellman = reward + gamma*Q_value_1_mean
            
            # print(type(Q_value_1_mean))
            # print(Q_value_1_mean)
            # print(type(reward))
            
            # print(type(Q_value_bellman))
            # print(Q_value_bellman)

            # update Q_value
            target = (1-alpha)*Q_value_0 + alpha*Q_value_bellman
            self.put_Q_value(observation_t0, action, target)
            
            action = self.pick_action(observation_t1, epsilon=epsilon_arg)

            observation_t0 = observation_t1

            if terminated or turncated:
                break

            if i>iter_limit:
                break
        
            i = i+1

        print("total steps : {}".format(i))

        return

    def drive_sarsa_tabel(self, epsilon_arg):
        ation_value_temp = np.empty(4)
        observation_t0, info = self.env.reset()

        action_t0 = self.pick_action(observation_t0, epsilon=epsilon_arg)

        
        # sarsa hyper parameter
        gamma = 0.9
        alpha = 0.8

        iter_limit = 1000
        i = 0
        while 1:
            Q_value_0 = self.get_Q_value(observation_t0, action_t0)
            observation_t1, reward, terminated, turncated, info = self.env.step(action_t0)

            action_t1 = self.pick_action(observation_t1, epsilon=epsilon_arg)

            Q_value_1 = self.get_Q_value(observation_t1, action_t1)

            # clac Q_value target
            if terminated or turncated:
                if terminated:
                    reward = 1000
                print("===========================")
                print(reward)
                Q_value_bellman = reward
            else:
                Q_value_bellman = reward + gamma*Q_value_1
            
            # print(type(Q_value_1_mean))
            # print(Q_value_1_mean)
            # print(type(reward))
            
            # print(type(Q_value_bellman))
            # print(Q_value_bellman)

            # update Q_value
            target = (1-alpha)*Q_value_0 + alpha*Q_value_bellman
            self.put_Q_value(observation_t0, action_t0, target)
            
            action_t0 = action_t1

            observation_t0 = observation_t1

            if terminated or turncated:
                break

            if i>iter_limit:
                break
        
            i = i+1

        print("total steps : {}".format(i))

        return

    def drive_by_key(self):
        
        observation, info = self.env.reset()
        while 1:
            Q_value = self.get_Q_value_all(observation)
            x, y = self.get_observation_to_xy(observation)
            print(x, y)
            print(Q_value)

            print("11111111111111111")
            action = self.get_keyboard_input()
            print(action)
            print("22222222222222222")
            if action == -1:
                break

            observation, reward, terminated, turncated, info = self.env.step(action)
            print(reward)


            if terminated or turncated:
                self.env.reset()

    def get_observation_to_xy(self, observation_arg):
        position_x = int(observation_arg%12)
        position_y = int(3-(observation_arg-observation_arg%12)/12)
        return position_x, position_y

    def get_Q_value(self, observation_arg, action_arg):
        position_x, position_y = self.get_observation_to_xy(observation_arg)
        return self.Q_table[position_x, position_y, action_arg]
    
    def get_Q_value_all(self, observation_arg):# a function which returns Qvalues for each every actions 
        ation_value_temp = np.empty(4)
        for i in range(4):
            ation_value_temp[i] = self.get_Q_value(observation_arg, i)
        return ation_value_temp

    def put_Q_value(self, observation_arg, action_arg, value_arg):
        position_x, position_y = self.get_observation_to_xy(observation_arg)
        self.Q_table[position_x, position_y, action_arg] = value_arg

    

    def pick_action(self, observation_arg, epsilon):
        #return type : int
        ation_value_all = self.get_Q_value_all(observation_arg)
        action_mask = self.get_action_mask(observation_arg)
        action_space = np.array(range(4))

        action_probability = action_mask/(action_mask.sum())
        # print("=====================")
        # print(action_mask.shape)
        # print(action_space.shape)
        # print(type(action_space))
        # print(type(action_space[0]))
        # print(action_probability)
        # print(action)

        if np.random.rand() < epsilon:
            # print("epslion_action")
            action_probability = action_mask/(action_mask.sum())
            action = np.random.choice(action_space, 1, p=action_probability)[0]#return type : int
        else:
            # print("greedy_action")
            ation_value_tabel = []

            max_value_temp = 0
            max_index_temp = -1
            i = 0
            n = 0
            for i in range(4):
                
                if action_mask[i] == 1:
                    if(max_index_temp == -1):
                        max_value_temp = ation_value_all[i]
                        max_index_temp = i
                    else:
                        if ation_value_all[i] > max_value_temp:
                            max_value_temp = ation_value_all[i]
                            max_index_temp = i
                        elif ation_value_all[i] == max_value_temp:
                            index = np.random.choice([max_index_temp, i], 1)[0]# to int
                            max_value_temp = ation_value_all[index]
                            max_index_temp = index

            if max_index_temp == -1:
                assert 0, "action mask error : all masks are 0"

            action_index = max_index_temp
            action = action_index
        
            # print("--------------")

        return action

    def get_action_mask(self, observation_arg):
        action_mask = np.ones(4)
        position_x, position_y = self.get_observation_to_xy(observation_arg)
        if position_x <= 0:
            action_mask[3] = 0
        if position_x >= 11:
            action_mask[1] = 0
        if position_y <= 0:
            action_mask[2] = 0
        if position_y >= 3:
            action_mask[0] = 0

        return action_mask

    def get_keyboard_input(self):
        key_input = -1

        key_value = input()
        # if key_value == "1":
        #     key_input = 0
        # elif key_value == "4":
        #     key_input = 1
        # elif key_value == "2":
        #     key_input = 2
        # elif key_value == "3":
        #     key_input = 3
        print(key_value)
        if key_value == "0":
            key_input = 0
        elif key_value == "1":
            key_input = 1
        elif key_value == "2":
            key_input = 2
        elif key_value == "3":
            key_input = 3

        # if key_input == -1:
        #     assert key_input==-1 , "wrong key input"

        return key_input


if __name__ == "__main__":
    env_screen = gym.make("CliffWalking-v0", render_mode="human")
    env_headless = gym.make("CliffWalking-v0")
    agent = Agent()

    agent.put_gym_env(env_headless)
    agent.drive_sarsa_tabel(1)
    for i in range(100):
        print("generation : {}".format(i))
        agent.drive_sarsa_tabel(0.1)
    
    agent.put_gym_env(env_screen)
    for i in range(100):
        agent.drive_sarsa_tabel(0)



    print("++++++++++++++++++++++++")
    env_screen.reset()
    time.sleep(3)
    for i in range(1000):
        agent.drive_by_key()
    print("++++++++++++++++++++++++")

    time.sleep(3)

    env_screen.close()
    env_headless.close()
