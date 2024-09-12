import csv
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
def learn(states, actions):
    states = np.array(states)
    actions = np.array(actions)
    lam = 1e-6
    demo = 3
    I = np.identity(demo)

    learned_thetea = actions @ (states.T) @ (np.linalg.inv(states @ states.T + lam * I))
    return learned_thetea

target_paths = [r'D:\KCL\year2\myPaper\icra\exp2\exp2_data\target_user_01',
                r'D:\KCL\year2\myPaper\icra\exp2\exp2_data\target_user_02',
                r'D:\KCL\year2\myPaper\icra\exp2\exp2_data\target_user_03',
                r'D:\KCL\year2\myPaper\icra\exp2\exp2_data\target_user_04',
                r'D:\KCL\year2\myPaper\icra\exp2\exp2_data\target_user_05',
                r'D:\KCL\year2\myPaper\icra\exp2\exp2_data\target_user_06',
                r'D:\KCL\year2\myPaper\icra\exp2\exp2_data\target_user_07',
                r'D:\KCL\year2\myPaper\icra\exp2\exp2_data\target_user_08',
                r'D:\KCL\year2\myPaper\icra\exp2\exp2_data\target_user_09',
                r'D:\KCL\year2\myPaper\icra\exp2\exp2_data\target_user_10',
                r'D:\KCL\year2\myPaper\icra\exp2\exp2_data\target_user_11',
                r'D:\KCL\year2\myPaper\icra\exp2\exp2_data\target_user_12',
                r'D:\KCL\year2\myPaper\icra\exp2\exp2_data\target_user_13',
                r'D:\KCL\year2\myPaper\icra\exp2\exp2_data\target_user_14',
                r'D:\KCL\year2\myPaper\icra\exp2\exp2_data\target_user_15',
                r'D:\KCL\year2\myPaper\icra\exp2\exp2_data\target_user_16']
                
control_paths = [r'D:\KCL\uArmSwiftPro\uArm-Python-SDK\real_exp_results\exp2_data\control_group_user_01',
                 r'D:\KCL\uArmSwiftPro\uArm-Python-SDK\real_exp_results\exp2_data\control_group_user_02',
                 r'D:\KCL\uArmSwiftPro\uArm-Python-SDK\real_exp_results\exp2_data\control_group_user_03',
                 r'D:\KCL\uArmSwiftPro\uArm-Python-SDK\real_exp_results\exp2_data\control_group_user_04',
                 r'D:\KCL\uArmSwiftPro\uArm-Python-SDK\real_exp_results\exp2_data\control_group_user_05',
                 r'D:\KCL\uArmSwiftPro\uArm-Python-SDK\real_exp_results\exp2_data\control_group_user_06',
                 r'D:\KCL\uArmSwiftPro\uArm-Python-SDK\real_exp_results\exp2_data\control_group_user_07',
                 r'D:\KCL\uArmSwiftPro\uArm-Python-SDK\real_exp_results\exp2_data\control_group_user_08',
                 r'D:\KCL\uArmSwiftPro\uArm-Python-SDK\real_exp_results\exp2_data\control_group_user_09',
                 r'D:\KCL\uArmSwiftPro\uArm-Python-SDK\real_exp_results\exp2_data\control_group_user_13',
                 r'D:\KCL\uArmSwiftPro\uArm-Python-SDK\real_exp_results\exp2_data\control_group_user_14',
                 r'D:\KCL\uArmSwiftPro\uArm-Python-SDK\real_exp_results\exp2_data\control_group_user_15',
                 r'D:\KCL\uArmSwiftPro\uArm-Python-SDK\real_exp_results\exp2_data\control_group_user_16',
                 r'D:\KCL\uArmSwiftPro\uArm-Python-SDK\real_exp_results\exp2_data\control_group_user_10',
                 r'D:\KCL\uArmSwiftPro\uArm-Python-SDK\real_exp_results\exp2_data\control_group_user_11',
                 r'D:\KCL\uArmSwiftPro\uArm-Python-SDK\real_exp_results\exp2_data\control_group_user_12']

target_el2s = []
control_el2s = []

for i in range(len(target_paths)):
    data = json.load(open(target_paths[i]))  # 读取每个文件的数据
    el2_list = []
    for j in range(12):
        if j == 2:
            phi = np.array(data['datas'][j]['phi'])  # 使用data而不是user_data
            user_action = np.array(data['datas'][j]['user_force'])
            learned_theta = learn(phi, user_action)
            real_theta = np.array(data['datas'][j]['real_theta'])
            el2 = np.linalg.norm(real_theta - learned_theta)
            el2_list.append(el2)
        else:
            el2_list.append(data['datas'][j]['el2'])    
    target_el2s.append(el2_list)

for i in range(len(control_paths)):
    data = json.load(open(control_paths[i]))  # 读取每个文件的数据
    el2_list = []
    for j in range(12):
        el2_list.append(data['datas'][j]['el2'])
    control_el2s.append(el2_list)

target_el2s = np.array(target_el2s)
control_el2s = np.array(control_el2s)

phases = ['P1E1', 'P2E1', 'P3E1', 'P3E2', 'P3E3', 'P3E4', 'P3E5', 'P3E6', 'P3E7', 'P3E8', 'P4E1', 'P5E1']
users = [f'{i:02}' for i in range(1, 17)]



##############################################
#get all thetas

target_thetas = []
target_real_thetas = []

control_thetas = []
control_real_thetas = []

for i in range(len(target_paths)):
    data = json.load(open(target_paths[i]))  # 读取每个文件的数据
    theta_list = []
    real_theta_list = []
    for j in range(12):
        if j == 2:
            phi = np.array(data['datas'][j]['phi'])  # 使用data而不是user_data
            user_action = np.array(data['datas'][j]['user_force'])
            learned_theta = learn(phi, user_action)
            real_theta = np.array(data['datas'][j]['real_theta'])

            theta_list.append(learned_theta)
            real_theta_list.append(real_theta)
            
        else:
            theta_list.append(data['datas'][j]['learned_theta'])
            real_theta_list.append(data['datas'][j]['real_theta'])

    target_thetas.append(theta_list)
    target_real_thetas.append(real_theta_list)

for i in range(len(control_paths)):
    data = json.load(open(control_paths[i]))  # 读取每个文件的数据
    theta_list = []
    real_theta_list = []
    for j in range(12):
        theta_list.append(data['datas'][j]['learned_theta'])
        real_theta_list.append(data['datas'][j]['real_theta'])

    control_thetas.append(theta_list)
    control_real_thetas.append(real_theta_list)


target_thetas = np.array(target_thetas)
control_thetas = np.array(control_thetas)

# np.save('exp2_target_thetas.npy', target_thetas)
# np.save('exp2_control_thetas.npy', target_thetas)


target_real_thetas = np.array(target_real_thetas)
control_real_thetas = np.array(control_real_thetas)
print("target_real_thetas", np.shape(control_thetas), np.shape(control_real_thetas))





# tihua
# 模拟系统的函数
def simulate_system(A, initial_state, steps=50):
    # x, y = initial_state[0], initial_state[1]
    x = []
    y = []
    delta_x = []
    delta_y = []

    current_state = np.array([  [initial_state[0]], 
                                [initial_state[1]], 
                                [1]])
    for _ in range(steps):
        x.append(current_state[0])
        y.append(current_state[1])

        # next_state = A @ np.array([x[0], y[-1], 1])
        delta_state = A @ current_state

        x_pos = current_state[0, 0] + delta_state[0, 0]
        y_pos = current_state[1, 0] + delta_state[1, 0]
        current_state = np.array([  [x_pos], 
                                    [y_pos], 
                                    [1]])

        delta_x.append(delta_state[0])
        delta_y.append(delta_state[1])
        if x_pos < -50 or x_pos > 350 or y_pos < -200 or y_pos > 200:
            break
    return x, y, delta_x, delta_y

# # 初始状态
initial_state = np.array([200, 20])
initial_state = np.array([30, -150])

x_range = np.linspace(-50, 350, 100)
y_range = 0.5 * x_range  
row_titles = ['User01', 'User02', 'User03', 'User04']
# 绘制轨迹
fig, axes = plt.subplots(4, 12, figsize=(15, 5), sharex=True, sharey=True)
for index, num in enumerate([0, 2]):
    for j in range(12):
        x_learned, y_learned, _, _ = simulate_system(target_thetas[num, j, : , :], initial_state)
        x_desired, y_desired, _, _ = simulate_system(target_real_thetas[num, j, : , :], initial_state)

        # axes[0, j].plot(x_learned, y_learned, marker='o', markersize=1.2, color = 'green', linewidth=1, linestyle='-')
        # axes[0, j].plot(x_desired, y_desired, marker='o', markersize=1.2, color = 'orange', linewidth=1, linestyle='-')
        axes[index, j].plot(x_learned, y_learned, color = 'green', linewidth=2, linestyle='-')
        axes[index, j].plot(x_desired, y_desired, color = 'orange', linewidth=2, linestyle='--')
        axes[index, j].scatter(x_desired[0], y_desired[0], marker='o', linewidths=2, color='green')
        axes[index, j].scatter(x_learned[-1], y_learned[-1], marker='o', linewidths=1, color='orange', alpha=0.4)
        if j == 1 or j == 11: 
            axes[index, j].plot(x_range, y_range, color='red', linestyle='--', linewidth=2)
        else:
            axes[index, j].scatter(x_desired[-1], y_desired[-1], marker='*', linewidths=2, color='red')
        
        # axes[0, j].axhline(0, color='black', linewidth=0.5)
        # axes[0, j].axvline(0, color='black', linewidth=0.5)
        axes[index, j].set_aspect('equal', 'box')
        axes[index, j].set_xlim(-50, 350)
        axes[index, j].set_ylim(-200, 200)
    axes[index, 0].set_ylabel(row_titles[index], size='large')


for index, num in enumerate([5, 2]):
    index = 2 + index
    for j in range(12):
        x_learned, y_learned, _, _ = simulate_system(control_thetas[num, j, : , :], initial_state)
        x_desired, y_desired, _, _ = simulate_system(control_real_thetas[num, j, : , :], initial_state)

        # axes[1, j].plot(x_learned, y_learned, marker='o', markersize=1.2, color = 'green', linewidth=1, linestyle='-')
        # axes[1, j].plot(x_desired, y_desired, marker='o', markersize=1.2, color = 'orange', linewidth=1, linestyle='-')
        axes[index, j].plot(x_learned, y_learned, color = 'green', linewidth=2, linestyle='-')
        axes[index, j].plot(x_desired, y_desired, color = 'orange', linewidth=2, linestyle='--')
        axes[index, j].scatter(x_desired[0], y_desired[0], marker='o', linewidths=2, color='green')
        axes[index, j].scatter(x_learned[-1], y_learned[-1], marker='o', linewidths=1, color='orange', alpha=0.4)
        if j == 1 or j == 11: 
            axes[index, j].plot(x_range, y_range, color='red', linestyle='--', linewidth=2)
        else:
            axes[index, j].scatter(x_desired[-1], y_desired[-1], marker='*', linewidths=2, color='red')

        # axes[1, j].axhline(0, color='black', linewidth=0.5)
        # axes[1, j].axvline(0, color='black', linewidth=0.5)
        axes[index, j].set_aspect('equal', 'box')
        axes[index, j].set_xlim(-50, 350)
        axes[index, j].set_ylim(-200, 200)
    axes[index, 0].set_ylabel(row_titles[index], size='large')

phase_lists = ['P1', 'P2', 'P3-1', 'P3-2', 'P3-3', 'P3-4', 'P3-5', 'P3-6', 'P3-7', 'P3-8', 'P4', 'P5']
for i in range(12):
    axes[index, i].set_xlabel(phase_lists[i], size='large')


# plt.tight_layout()
plt.subplots_adjust(wspace=-0.2, hspace=0.13) 
plt.show()
