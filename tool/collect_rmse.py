import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json
# workspace: x: 0-350 y:-250-250 z:0
# Setup the uArm
sys.path.append(os.path.join(os.path.dirname(__file__), 'c:/Users/11424/OneDrive/文档/uArm-Python-SDK'))
from uarm.wrapper import SwiftAPI
swift = SwiftAPI(port="COM3", callback_thread_pool_size=1)

# def get_starting_points():
#     starting_points = []
#     for i in range(5):
#         test_starting_points = np.ones(3)
#         test_starting_points[0] = int(np.random.uniform(0, 350))
#         test_starting_points[1] = int(np.random.uniform(-250, 250))
#         test_starting_points[2] = 0
#         starting_points.append(test_starting_points.tolist())
#     return starting_points


def plot_trajectory(trajectory):
    plt.figure()
    for i in range(5):
        traj = np.array(trajectory[i])
        plt.plot(traj[:, 0], traj[:, 1])
    plt.show()

def experiments(learned_theta, test_starting):
    trajs = []
    print(test_starting)
    for i in range(5):
        print(f"----------------Trajectory {i+1}--------------")
        test_starting_points = test_starting[i]
        # Student Demonstration
        swift.reset(x=test_starting_points[0], y=test_starting_points[1], z=0)
        # test_point = np.array([test_starting_points[0], test_starting_points[1] + 20]) # test_starting_points = [130, 60, 0]
        
        x, y, z = swift.get_position(timeout=100000)
        vel = 1000
        
        steps = 0
        trajectory = []
        while steps < 70:
            trajectory.append([x, y])

            delta_x, delta_y = learned_theta @ np.array([x, y, 1]).T
            x += delta_x
            y += delta_y
            swift.set_position(x=x, y=y, z=0, speed=vel*60)
            
            x, y, z = swift.get_position(timeout=100000)
            steps += 1
        trajs.append(trajectory)
    return trajs
        
def result_collection(thetas):
    
    for i in range(len(thetas)):
        print(f"----------------Person {i+1}--------------")
        result = defaultdict(list)
        filename = rf'C:\Users\11424\OneDrive\文档\phd\TwoLinkRobotArmSimulation\tool\target_results_{i+1}.json'
        # number of people
        for j in range(len(thetas[0])):
            print(f"----------------Phase {j+1}--------------")
            trajs = experiments(thetas[i][j], starts)
            result[f"phase_{j+1}"].append(trajs)
                # Save to JSON file
        with open(filename, 'w') as f:
            json.dump(result, f)

starts = [[142, -179, 0.0], [238, -160, 0.0], [215, 17, 0.0], [63, -216, 0.0], [105, 157, 0.0]]

control_thetas =  np.load(rf'C:\Users\11424\Downloads\exp2_control_thetas.npy')
target_thetas = np.load(rf'C:\Users\11424\Downloads\exp2_target_thetas.npy')

# print(len(target_thetas))
# print(len(target_thetas[0]))
# trajs = experiments(control_thetas[0][0], starts)
# print(trajs)
# plot_trajectory(trajs)
result_collection(target_thetas)
