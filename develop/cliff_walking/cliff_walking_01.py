import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


def env_drive_by_key(env_arg):
    env.reset()
    while (1):
        print("==================")
        key_input = get_keyboard_input()
        print(key_input) 
        if key_input == 4:
            break

        observation, reward, terminated, truncated, info = env_arg.step(key_input)
        print(observation)
        print(reward)
        print(terminated)
        print(truncated)
        
        if terminated == 1:
            break

def get_keyboard_input():
    key_input = -1
    # while 1:
    #     if keyboard.is_pressed("up") == 1:
    #         key_input = 0
    #     if keyboard.is_pressed("right") == 1:
    #         key_input = 1
    #     if keyboard.is_pressed("down") == 1:
    #         key_input = 2
    #     if keyboard.is_pressed("left") == 1:
    #         key_input = 3

    #     if keyboard.is_pressed("x") == 1:
    #         key_input = 4
    #     if keyboard.is_pressed("X") == 1:
    #         key_input = 4
    #     if keyboard.is_pressed("esc") == 1:
    #         key_input = 4

        # if key_input != -1: 
        #     break

    key_value = input()
    if key_value == "1":
        key_input = 0
    elif key_value == "4":
        key_input = 1
    elif key_value == "2":
        key_input = 2
    elif key_value == "3":
        key_input = 3


    return key_input

# 0: LEFT
# 1: DOWN
# 2: RIGHT
# 3: UP

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
    
    def drive_sarsa_tabel(self, epsilon_arg):
        ation_value_temp = np.empty(4)
        observation, info = self.env.reset()

        action_t0 = self.pick_action(observation, epsilon=epsilon_arg)

        Q_value_0 = self.get_Q_value(observation, action_t0)
        # sarsa hyper parameter
        gamma = 0.99
        alpha = 0.7

        iter_limit = 100000
        i = 0
        while 1:
            observation, reward, terminated, turncated, info = self.env.step(action_t0)

            action_t1 = self.pick_action(observation, epsilon=epsilon_arg)

            Q_value_1 = self.get_Q_value(observation, action_t1)

            print(Q_value_0)
            # clac Q_value target
            if terminated or turncated or (reward == -100):
                Q_value_bellman = reward
            else:
                Q_value_bellman = reward + gamma*Q_value_1

            # update Q_value
            target = (1-alpha)*Q_value_0 + alpha*Q_value_bellman
            self.put_Q_value(observation, action_t0, target)

            action_t0 = action_t1

            if reward == -100:
                break
            
            if terminated or turncated:
                break

            if i>iter_limit:
                break
        
            i = i+1

        return
    
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
            print("epslion_action")
            action_probability = action_mask/(action_mask.sum())
            action = np.random.choice(action_space, 1, p=action_probability)[0]#return type : int
        else:
            print("greedy_action")
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

            if max_value_temp == -1:
                assert 0, "action mask error : all masks are 0"

            action_index = max_index_temp
            action = action_index
        


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




if __name__ == "__main__":
    env_screen = gym.make("CliffWalking-v0", render_mode="human")
    env_headless = gym.make("CliffWalking-v0")


    # env.reset()
    # env_drive_by_key(env)
    
    agent = Agent()
    # agent.put_gym_env(env_headless)
    agent.put_gym_env(env_screen)

    # agent.put_Q_value(36, 0, 10)
    # agent.put_Q_value(36, 1, 100)
    # agent.put_Q_value(36, 2, -100)
    # agent.put_Q_value(36, 3, -100)

    # print(agent.get_Q_value(36, 0))
    # print(agent.get_Q_value(36, 3))
    
    # print(agent.pick_action(36, 0))
    # print(agent.pick_action(36, 0))
    # print(agent.pick_action(36, 1))
    # print(agent.pick_action(36, 1))
    # print(agent.pick_action(36, 1))
    # print(agent.pick_action(36, 1))
    # print(agent.pick_action(36, 1))

    agent.drive_sarsa_tabel(1)
    for i in range(100):
        agent.drive_sarsa_tabel(0.2)
    
    agent.put_gym_env(env_screen)
    agent.drive_sarsa_tabel(0.2)
    agent.drive_sarsa_tabel(0.1)
    agent.drive_sarsa_tabel(0.05)
    agent.drive_sarsa_tabel(0)

    print("++++++++++++++++++++++++")
    observation_arg = 36
    print(agent.get_observation_to_xy(observation_arg))
    print(agent.get_action_mask(observation_arg))
    print(agent.pick_action(observation_arg, 0))

    time.sleep(3)

    env_screen.close()
    env_headless.close()
