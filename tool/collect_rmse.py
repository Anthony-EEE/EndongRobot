import os
import sys
import json
import math

import pyttsx3
import keyboard
import numpy as np
import time
from sklearn.preprocessing import normalize

test_starting_points = [280, -30, 0]

# Setup the uArm
sys.path.append('..')
from uarm.wrapper import SwiftAPI
swift = SwiftAPI(port="COM8", callback_thread_pool_size=1)
print(swift.get_device_info())

def learn(states, actions, demo=5, lam=1e-6):
    lam = lam
    demo = 3
    I = np.identity(demo)

    learned_thetea = actions @ (states.T) @ (np.linalg.inv(states @ states.T + lam * I))
    return learned_thetea


def experiments(elements, target_skill, score, real_theta, states, real_actions):
    print("states", states)
    user_actions = []


    user_actions = np.array(user_actions).T
    learned_theta = learn(states, user_actions, demo=3)
    el2 = np.linalg.norm(real_theta - learned_theta)



    # Student Demonstration
    test_point = np.array([test_starting_points[0], test_starting_points[1] + 20]) # test_starting_points = [130, 60, 0]
    swift.reset(x=test_starting_points[0], y=test_starting_points[1], z=50)
    
    x, y, z = swift.get_position(timeout=100000)
    vel = 100  # 固定速度 mm/second
    vel = 50
    steps = 0
    trajectory = []
    while steps < 50:
        trajectory.append([x, y])

        delta_x, delta_y = learned_theta @ np.array([x, y, 1]).T
        x += delta_x
        y += delta_y
        swift.set_position(x=x, y=y, z=50, speed=vel*60)
        
        x, y, z = swift.get_position(timeout=100000)
        steps += 1
    diff = true_traj - trajectory
    distances = np.linalg.norm(diff, axis=1)
    E_RMSE = np.mean(distances)
        


user_id = 'control_group_user_13'

s1_desired_traj = np.load('skill_1_real_trajectory.npy')
s2_desired_traj = np.load('skill_2_real_trajectory.npy')


# s1_state = np.array([[320, 220, 140],
#                      [50, 170, -30],
#                      [1, 1, 1]])
s2_state = np.array([[280.0, 260., 130.],
                     [60., -50., 70.],
                     [1, 1, 1]])

s1_state = np.array([[300, 220, 200],
                     [60, 140, -70],
                     [1, 1, 1]])

s2_state_prime = np.array([ [320, 150, 250],
                            [30, -20, 150],
                            [1, 1, 1]])




# # Real controller
s1_theta = np.array([[-0.04, 0.08, 0.0],
                       [0.08, -0.16, 0.0]])#点到斜线y=0.5x
s2_theta = np.array([[-0.2, 0, 0.2*230],
                       [0, -0.2, 0.2*110]]) 

s1_actions = s1_theta @ s1_state
s2_actions = s2_theta @ s2_state
s2_actions_prime = s2_theta @ s2_state_prime

