import os
import csv
import math
import datetime
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import tkinter.messagebox as messagebox
from sklearn.preprocessing import normalize
import matplotlib.image as mpimg
from matplotlib.transforms import Affine2D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json

import numpy as np

from SingleArm import SingleRobotArm

class Student:
    def __init__(self, robot, x0,y0,xd,yd):
        self.robot = robot
        self.x0 = x0
        self.y0 = y0
        self.xd = xd
        self.yd = yd

    def reset(self, x0, y0, xd, yd):
        self.x0 = x0
        self.y0 = y0
        self.xd = xd
        self.yd = yd
        self.robot.reset()

    def student_demostration(self, plot_animation_flag, update_step, L_learnable):
        self.robot.joint1_angle = self.robot.inverse_kinematics(self.x0, self.y0)        
        self.robot.set_L(L_learnable)

        for i in range(update_step):
            self.robot.trajectory.append(list(self.robot.get_end_effector_position()))
            self.robot.velocity.append(list(self.robot.get_end_effector_velocity()))
            
            if plot_animation_flag:
                # Plot the current state of the robot arm and the trajectory
                plt.clf()
                plt.xlim(self.robot.x_min, self.robot.x_max)
                plt.ylim(self.robot.y_min, self.robot.y_max)
                plt.plot(self.x0, self.y0, marker="o", markersize=3, markeredgecolor="cyan", markerfacecolor="cyan", label='Starting Point')
                plt.plot(self.xd, self.yd, marker="o", markersize=3, markeredgecolor="green", markerfacecolor="green", label='Target Point')
                plt.legend(loc='upper left')
                plt.plot([0, self.robot.link1_length * np.cos(self.robot.joint1_angle), self.robot.link1_length * np.cos(self.robot.joint1_angle) ], 
                            [0, self.robot.link1_length * np.sin(self.robot.joint1_angle), self.robot.link1_length * np.sin(self.robot.joint1_angle)],
                            '-o', color = 'blue')
                plt.plot([pos[0] for pos in self.robot.trajectory], [pos[1] for pos in self.robot.trajectory], '-r')
                plt.grid()
                plt.gca().set_aspect("equal")
                plt.draw()
                plt.title('Student: Frame {}'.format(i+1))
                plt.pause(0.001) 
            
            # robot.update() 
            # self.robot.update_rk4() # updates using RK4 method
            self.robot.update()
        self.trajectory = self.robot.trajectory
        self.velocity = self.robot.velocity    

    def plot_dynamic_motion(self, mode):
        # Create a figure with 4 subplots
        trajectory = np.array(self.robot.trajectory)
        velocity = np.array(self.robot.velocity)
        
        fig, axs = plt.subplots(4, 2)
        if mode == 1:
            fig.suptitle("Student's Dynamics Motion, teaching all points, real force")
        elif mode == 2:
            fig.suptitle("Student's Dynamics Motion, teaching all points, noise force")
        elif mode == 3:
            fig.suptitle("Student's Dynamics Motion, teaching 4 random selected points, noise force")
        
        #fig.suptitle("Student's Dynamics Motion")
        #fig.suptitle('L = {}\n       {}'.format(self.robot.L[0], self.robot.L[1]))

        idx = [i for i in range(len(self.robot.joint1_angle_track))]
        
        # Plot the lines on the subplots
        axs[0, 0].plot(idx, self.robot.joint1_angle_track)
        axs[0, 1].plot(idx, self.robot.joint2_angle_track)
        axs[1, 0].plot(idx, self.robot.joint1_velocity_track)
        axs[1, 1].plot(idx, self.robot.joint2_velocity_track)
        axs[2, 0].plot(idx, trajectory[:, 0])
        axs[2, 1].plot(idx, trajectory[:, 1])
        axs[3, 0].plot(idx, velocity[:, 0])
        axs[3, 1].plot(idx, velocity[:, 1])

        # Add titles to the subplots
        axs[0, 0].set_title('q1')
        axs[0, 1].set_title('q2')
        axs[1, 0].set_title('q1_dot')
        axs[1, 1].set_title('q2_dot')
        axs[2, 0].set_title('x')
        axs[2, 1].set_title('y')
        axs[3, 0].set_title('x_dot')
        axs[3, 1].set_title('y_dot')

        # Adjust the spacing between the subplots
        fig.tight_layout()

        # Show the plot
        plt.show()

class Teacher:
    def __init__(self, robot, x0,y0,xd,yd, noise_mean=0, noise_std=0.0):
        self.robot = robot
        self.x0 = x0
        self.y0 = y0
        self.xd = xd
        self.yd = yd
        self.trajectory = []
        self.velocity = []

        # set noise level for teacher robot
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.robot.set_noise_level(noise_mean, noise_std) 

        self.all_demo_comb = 0
        self.points_num = 0

    def reset(self, x0, y0, xd, yd):
        self.x0 = x0
        self.y0 = y0
        self.xd = xd
        self.yd = yd
        
        self.robot.reset()

    def teacher_demonstration(self, plot_animation_flag, update_step, L_kd):
        self.robot.joint1_angle = self.robot.inverse_kinematics(self.x0, self.y0)

        L = np.array(L_kd)
        self.robot.set_L(L)
        #self.set_L(L=np.array([[-1, 0, -1, 0, (-1)*(-xd), 0], [0, -1, 0, -1, 0, (-1)*(-yd)]]))

        for i in range(update_step):
            self.robot.trajectory.append(list(self.robot.get_end_effector_position()))
            self.robot.velocity.append(list(self.robot.get_end_effector_velocity()))
            
            if plot_animation_flag:
                # Plot the current state of the robot arm and the trajectory
                plt.clf()
                plt.xlim(self.robot.x_min, self.robot.x_max)
                plt.ylim(self.robot.y_min, self.robot.y_max)
                plt.plot(self.x0, self.y0, marker="o", markersize=3, markeredgecolor="cyan", markerfacecolor="cyan", label='Starting Point')
                plt.plot(self.xd, self.yd, marker="o", markersize=3, markeredgecolor="green", markerfacecolor="green", label='Target Point')
                plt.legend(loc='upper left')
                # plt.plot([0, self.robot.link1_length * np.cos(self.robot.joint1_angle), self.robot.link1_length * np.cos(self.robot.joint1_angle)], 
                #             [0, self.robot.link1_length * np.sin(self.robot.joint1_angle), self.robot.link1_length * np.sin(self.robot.joint1_angle)],
                #             '-o', color = 'blue')

                plt.plot([0, self.robot.link1_length * np.cos(self.robot.joint1_angle)], 
                            [0, self.robot.link1_length * np.sin(self.robot.joint1_angle)],
                            '-o', color = 'blue')

                plt.plot([pos[0] for pos in self.robot.trajectory], [pos[1] for pos in self.robot.trajectory], '-r')
                plt.grid()
                plt.gca().set_aspect("equal")
                plt.draw()
                plt.title('Teacher: Frame {}'.format(i+1))
                plt.pause(0.001) 
            
            # robot.update() 
            # self.robot.update_rk4() # updates using RK4 method
            self.robot.update()

        self.trajectory = self.robot.trajectory
        self.velocity = self.robot.velocity

    def plot_dynamic_motion(self):
        # Create a figure with 4 subplots
        trajectory = np.array(self.robot.trajectory)
        velocity = np.array(self.robot.velocity)

        fig, axs = plt.subplots(4, 2)
        fig.suptitle("Teacher's Dynamics Motion")
        #fig.suptitle('Teacher\'s Dynamics Motion \n L = {}\n       {}'.format(self.robot.L[0], self.robot.L[1]))

        idx = [i for i in range(len(self.robot.joint1_angle_track))]
        
        # Plot the lines on the subplots
        axs[0, 0].plot(idx, self.robot.joint1_angle_track)
        axs[0, 1].plot(idx, self.robot.joint2_angle_track)
        axs[1, 0].plot(idx, self.robot.joint1_velocity_track)
        axs[1, 1].plot(idx, self.robot.joint2_velocity_track)
        axs[2, 0].plot(idx, trajectory[:, 0])
        axs[2, 1].plot(idx, trajectory[:, 1])
        axs[3, 0].plot(idx, velocity[:, 0])
        axs[3, 1].plot(idx, velocity[:, 1])

        # Add titles to the subplots
        axs[0, 0].set_title('q1')
        axs[0, 1].set_title('q2')
        axs[1, 0].set_title('q1_dot')
        axs[1, 1].set_title('q2_dot')
        axs[2, 0].set_title('x')
        axs[2, 1].set_title('y')
        axs[3, 0].set_title('x_dot')
        axs[3, 1].set_title('y_dot')

        # Adjust the spacing between the subplots
        fig.tight_layout()

        # Show the plot
        plt.show()
        # plt.pause(0.001) 

    def teaching(self, mode, S=4, teach_with_noise=True):
        """
        Mode 1: teaching all points with real force to the student
        Mode 2: teaching all points with Gaussian noise force to the student
        Mode 3: teaching selected 4 points with Gaussian noise force to the student
        """

        det = None

        real_force = np.squeeze(np.array(self.robot.real_force))
        noise_force = np.squeeze(np.array(self.robot.noise_force))
        phi = np.squeeze(np.array(self.robot.phi)).T

        if mode == 1:
            theta_learner = self.theta_fun(phi, real_force)
        elif mode == 2:
            theta_learner = self.theta_fun(phi, noise_force)
        elif mode == 3:
            index_list = [x for x in range(real_force.shape[0])]
            all_demo_combinations = list(itertools.combinations(index_list, S))

            selected_combination = random.choice(all_demo_combinations)
            selected_force = real_force[selected_combination,:] if not teach_with_noise else noise_force[selected_combination, :]
            selected_phi = phi[:,selected_combination]
            
            det = np.abs(np.linalg.det((np.array(selected_phi[0:4,:]))))

            theta_learner = self.theta_fun(selected_phi, selected_force)
        else:
            print("ERROR.")

        self.all_demo_comb = len(all_demo_combinations)
        self.points_num = real_force.shape[0]

        #print(self.points_num)
        #print(self.al
        return det, theta_learner

    def theta_fun(self, phi, force):
        lam = 1e-6
        I = np.identity(5)
        theta_learner = (np.linalg.inv(phi @ np.transpose(phi) + lam * I) @ phi @ force).T
        return theta_learner

if __name__ == '__main__':   
    # Robot configurations and inital states
    link1_length = 1
    link1_mass = 1
    joint1_angle = 0
    joint1_velocity = 0.0
    joint1_torque = 0.0
    time_step = 0.1
    g = 9.81
    teacher_animation_flag = True
    # initialize
    lam = 1e-6

    x0, y0 = 0.0, 1.0
    xd, yd = 1.0, 0.0
    update_step = 300
    L = np.array([3, 3])
    
    # intialize teacher robot and its noise level
    teacher_robot = SingleRobotArm(
        link1_length=link1_length, 
        link1_mass=link1_mass, 
        joint1_angle=joint1_angle, 
        joint1_velocity=joint1_velocity,
        joint1_torque=joint1_torque,
        time_step=time_step, g=g)
    
    
   
    teacher = Teacher(teacher_robot,x0, y0, xd, yd)
    teacher.teacher_demonstration(teacher_animation_flag, update_step=update_step, L_kd=L)
    # teacher.plot_dynamic_motion()
