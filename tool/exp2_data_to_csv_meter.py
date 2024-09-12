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
                
control_paths = [r'D:\KCL\uArmSwiftPro\uArm-Python-SDK\real_exp_results\exp2_data\control_group_user_02',
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
                 r'D:\KCL\uArmSwiftPro\uArm-Python-SDK\real_exp_results\exp2_data\control_group_user_01',
                 r'D:\KCL\uArmSwiftPro\uArm-Python-SDK\real_exp_results\exp2_data\control_group_user_01',
                 r'D:\KCL\uArmSwiftPro\uArm-Python-SDK\real_exp_results\exp2_data\control_group_user_01',
                 r'D:\KCL\uArmSwiftPro\uArm-Python-SDK\real_exp_results\exp2_data\control_group_user_01']

target_el2s = []
control_el2s = []

for i in range(len(target_paths)):
    data = json.load(open(target_paths[i]))  # 读取每个文件的数据
    el2_list = []
    for j in range(12):
        phi = np.array(data['datas'][j]['phi'], dtype=np.float64)  # 使用data而不是user_data
        user_action = np.array(data['datas'][j]['user_force'], dtype=np.float64) 
        print("phi", phi)
        phi[:2, :] /= 10.0
        user_action /= 10.0
        learned_theta = learn(phi, user_action)
        real_theta = np.array(data['datas'][j]['real_theta'])
        real_theta[:, 2] /= 10
        
        el2 = np.linalg.norm(real_theta - learned_theta)
        el2_list.append(el2)
    target_el2s.append(el2_list)


for i in range(len(control_paths)):
    data = json.load(open(control_paths[i]))  # 读取每个文件的数据
    el2_list = []
    for j in range(12):
        phi = np.array(data['datas'][j]['phi'], dtype=np.float64)  # 使用data而不是user_data
        user_action = np.array(data['datas'][j]['user_force'], dtype=np.float64) 
        print("phi", phi)
        phi[:2, :] /= 10.0
        user_action /= 10.0
        learned_theta = learn(phi, user_action)
        real_theta = np.array(data['datas'][j]['real_theta'])
        real_theta[:, 2] /= 10

        el2 = np.linalg.norm(real_theta - learned_theta)
        el2_list.append(el2)
    control_el2s.append(el2_list)


target_el2s = np.array(target_el2s)
control_el2s = np.array(control_el2s)



phases = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
print(np.shape(target_el2s), np.shape(target_el2s))
print(np.shape(control_el2s), np.shape(control_el2s))


p1_target = target_el2s[:, 0]  # target组的p1
p2_target = target_el2s[:, 1]
p3_target = target_el2s[:, 9]  
p4_target = target_el2s[:, 10]  # target组的p4
p5_target = target_el2s[:, 11]  

p1_control = control_el2s[:, 0]  # control组的p1
p2_control = control_el2s[:, 1]
p3_control = control_el2s[:, 9]
p4_control = control_el2s[:, 10]  # control组的p4
p5_control = control_el2s[:, 11]

p3p1_target = p3_target - p1_target
p3p1_control = p3_control - p1_control

p4p1_target = p4_target - p1_target
p4p1_control = p4_control - p1_control

p5p2_target = p5_target - p2_target
p5p2_control = p5_control - p2_control

print("p3p1_target", np.shape(p3p1_control), type(p3p1_target))



df = pd.DataFrame({
    'p3p1_target': p3p1_target,
    'p3p1_control': p3p1_control
})
df.to_csv(r'D:\KCL\year2\myPaper\icra\exp2\p3p1_data.csv', index=False)


df = pd.DataFrame({
    'p4p1_target': p4p1_target,
    'p4p1_control': p4p1_control
})
df.to_csv(r'D:\KCL\year2\myPaper\icra\exp2\p4p1_data.csv', index=False)

df = pd.DataFrame({
    'p5p2_target': p5p2_target,
    'p5p2_control': p5p2_control
})
df.to_csv(r'D:\KCL\year2\myPaper\icra\exp2\p5p2_data.csv', index=False)




# # data = {
# #     'p1_target': p1_target,
# #     'p1_control': p1_control,
# #     'p3_target': p3_target,
# #     'p3_control': p3_control
# # }
# # df = pd.DataFrame(data)
# # df.to_csv(r'D:\KCL\year2\myPaper\icra\exp1\p1p1p3p3_data.csv', index=False)
