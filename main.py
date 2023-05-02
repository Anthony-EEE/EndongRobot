import math
import random
import itertools
import matplotlib.pyplot as plt

import numpy as np

from two_link_robot import RobotArm
from learner import Learner

S = 4

# Initialize the robot arm object
robot_arm = RobotArm(link1_length=1.0, link2_length=1.0, link1_mass=1.0, link2_mass=1.0, 
                    joint1_angle=0.0, joint2_angle=0.0, joint1_velocity=0.0, joint2_velocity=0.0, joint1_torque=0.0, joint2_torque=0.0, 
                    time_step=0.01, ifdemo=True, g=0)

# initalize learner
learner = Learner(S=S, lamb=1e-6, link1_length=1.0, link2_length=1.0, link1_mass=1.0, link2_mass=1.0,
                 joint1_angle=0.0, joint2_angle=0.0, joint1_velocity=0.0, joint2_velocity=0.0, joint1_torque=0.0, joint2_torque=0.0, 
                 time_step=0.01, g=9.81)

def teacher_demonstration_move_to(desired_position, desired_velocity, stiffness=np.diag([5.0, 5.0]), damping=np.diag([1.0, 1.0]), threshold=[0.01, 0.5], max_iterations=1000, plot=True):
    robot_arm.move_to(desired_position, desired_velocity, stiffness=stiffness, damping=damping, threshold=threshold, max_iterations=max_iterations, plot=plot)

def teacher_draw_a_circle(center=[0.8, 0.8], radius=0.2, num_points = 100, stiffness=np.diag([5.0, 5.0]), damping=np.diag([1.0, 1.0]), drawing_accuracy_tolerance=[0.01, 0.5], max_iterations=1000):
    robot_arm.move_to_circle(center[0], center[1], radius, num_points=num_points, stiffness=stiffness, damping=damping, threshold=drawing_accuracy_tolerance, max_iterations=max_iterations)

def learner_learn_and_perform_once(demonstration_indexs, desired_position, desired_velocity, threshold=[0.01, 0.5], max_iterations=1000, plot=True):
    learner.learning_process([robot_arm.demonstrations[i] for i in demonstration_indexs])
    learner.perform_learnt_outcomes(desired_position, desired_velocity, threshold=threshold, max_iterations=max_iterations, plot=plot)

def learner_learn_and_circle(demonstration_indexs, inital_q1q2 = [0, 0], inital_q1dq2d = [2*np.pi, 0], plot=True):
    learner.learning_process([robot_arm.demonstrations[i] for i in demonstration_indexs])
    learner.learner_circle(inital_q1q2 = inital_q1q2, inital_q1dq2d = inital_q1dq2d, plot = plot)

def learnner_learn_and_perform_ntimes(num, teacher_traj, inital_q1q2 = [0, 0], inital_q1dq2d = [2*np.pi, 0], plot = False):
    index_list = [x for x in range(len(robot_arm.demonstrations))]

    combinations_of_four = list(itertools.combinations(index_list, S))

    print(len(combinations_of_four))

    traj_rmse = []
    for item in random.sample(combinations_of_four, num):
        # learner_learn_and_perform_once(demonstration_indexs=item, desired_position=desired_position, desired_velocity=desired_velocity, threshold=threshold, max_iterations=max_iterations, plot=plot)
        learner_learn_and_circle(demonstration_indexs=item, inital_q1q2 = inital_q1q2, inital_q1dq2d = inital_q1dq2d, plot = plot)

        traj_rmse_ = 0
        try: 
            for i in range(len(teacher_traj)):
                traj_rmse_ += np.sqrt((teacher_traj[i][0]-learner.trajectory[i][0])**2 + (teacher_traj[i][1]-teacher_traj[i][1])**2)
            traj_rmse_ = traj_rmse_/len(teacher_traj)
        except:
            traj_rmse_ = math.nan

        traj_rmse.append(traj_rmse_)

        learner.trajectory = []

    trajrmse = []
    det = []
    for i in range(len(traj_rmse)):
        if not math.isnan(traj_rmse[i]) and not math.isinf(traj_rmse[i]) and traj_rmse[i] < 10000:
            det.append(learner.det_phi[i])
            trajrmse.append(traj_rmse[i])

    plt.figure()
    plt.plot(det, trajrmse, 'o')
    plt.title('det_phi vs trajectory RMSE')
    plt.show()

if __name__ == '__main__':

    # TASK == 1: Robot draw a circle
    # TASK == 2: Student learn from a set of demonstration
    # TASK == 3: Sample N demonstration sets from demonstration space for Machine Teaching algorithm evaluation
    TASK = 4

    if TASK ==1:
        # Teacher draw a circle
        teacher_draw_a_circle()
    elif TASK == 2:
        demonstration_set_id = 100 # for example, good student id is 100; bad student id 10012125
        # Teacher demonstration
        teacher_demonstration_move_to(desired_position=[1.0, 1.0], desired_velocity=[0, 0], 
                                    stiffness=np.diag([200.0, 200.0]), damping=np.diag([10.0, 10.0]), 
                                    threshold=[0.01, 0.5], max_iterations=1000, plot=True)

        index_list = [x for x in range(len(robot_arm.demonstrations))]

        combinations_of_four = list(itertools.combinations(index_list, S))

        # Learner learning once from 4x4 PHI demonstration matrix
        learner_learn_and_perform_once(combinations_of_four[demonstration_set_id], desired_position=[1.0, 1.0], desired_velocity=[0, 0], 
                                    threshold=[0.01, 0.5], max_iterations=1000, plot=True)

        traj_rmse = 0
        try: 
            for i in range(len(robot_arm.trajectory)):
                traj_rmse += np.sqrt((robot_arm.trajectory[i][0]-learner.trajectory[i][0])**2 + (robot_arm.trajectory[i][1]-learner.trajectory[i][1])**2)
            traj_rmse = traj_rmse/len(robot_arm.trajectory)
        except:
            traj_rmse = math.nan

        print('For demonstration set {}, det_phi is {}, final position RMSE is {}, traj RMSE is {}.'.format(combinations_of_four[demonstration_set_id], learner.det_phi, learner.rmse, traj_rmse))
    elif TASK == 3:
        # Teacher demonstration
        teacher_demonstration_move_to(desired_position=[1.0, 1.0], desired_velocity=[0, 0], 
                                    stiffness=np.diag([5.0, 5.0]), damping=np.diag([1.0, 1.0]), 
                                    threshold=[0.01, 0.5], max_iterations=1000, plot=False)
        # Machine Teaching evaluation
        # Will plot det(phi) vs RMSE
        learnner_learn_and_perform_ntimes(num=10000, desired_position=[1.0, 1.0], desired_velocity=[0, 0], teacher_traj=robot_arm.trajectory,
                                    threshold=[0.01, 0.5], max_iterations=1000, plot=False)
    elif TASK == 4:
        robot_arm.circle(inital_q1q2 = [0, np.pi/8], inital_q1dq2d = [0, 0], plot=True)
        # print('Teacher: ', robot_arm.circle_L)

        # learner_learn_and_circle(demonstration_indexs=[1, 22, 3, 10], inital_q1q2 = [0, 0], inital_q1dq2d = [np.pi, 0], plot=True)
        # learnner_learn_and_perform_ntimes(num=1000, teacher_traj=robot_arm.trajectory, inital_q1q2 = [0, np.pi/4], inital_q1dq2d = [np.pi, 0], plot = False)
    else:
        pass