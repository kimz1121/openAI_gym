import time
import keyboard

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


def env_drive_by_key(env_arg):
    env.reset()
    while (1):
        print("==================")
        key_input = get_keyboard_input
        print(key_input) 
        if key_input == 4:
            break

        observation, reward, terminated, truncated, info = env_arg.step()
        print(observation)
        print(reward)
        print(terminated)
        print(truncated)
        
        if terminated == 1:
            break



def get_keyboard_input():
    key_input = -1
    while 1:
        if keyboard.is_pressed("up") == 1:
            key_input = 0
        if keyboard.is_pressed("right") == 1:
            key_input = 1
        if keyboard.is_pressed("down") == 1:
            key_input = 2
        if keyboard.is_pressed("left") == 1:
            key_input = 3

        if keyboard.is_pressed("x") == 1:
            key_input = 4
        if keyboard.is_pressed("X") == 1:
            key_input = 4
        if keyboard.is_pressed("esc") == 1:
            key_input = 4

        if key_input != -1: 
            break

    return key_input

# 0: LEFT
# 1: DOWN
# 2: RIGHT
# 3: UP


if __name__ == "__main__":
    env = gym.make("CliffWalking-v0", render_mode="human")



    env_drive_by_key(env)

    time.sleep(3)

    env.close()
