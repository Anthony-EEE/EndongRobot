import os
import sys
import csv
import json
import math
import datetime
import numpy as np
import tkinter as tk
import time

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import tkinter.messagebox as messagebox
from sklearn.preprocessing import normalize
import matplotlib.image as mpimg
from matplotlib.transforms import Affine2D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from BasicRobotArm import BasicRobotArm
import seaborn as sns
import datetime


# joint_image = mpimg.imread("./RobotConfigs/Joint.png")
# link_image = mpimg.imread("./RobotConfigs/Link.png")
# # Inside the class initialization
# end_effector_image = mpimg.imread("./RobotConfigs/EndEffector.png")
# base_image = mpimg.imread("./RobotConfigs/Base.png")


joint_image = mpimg.imread(rf'D:\KCL\year1\code\MTRobotSimulator_v2\v3\RobotConfigs\Joint.png')
link_image = mpimg.imread(rf'D:\KCL\year1\code\MTRobotSimulator_v2\v3\RobotConfigs\Link.png')
end_effector_image = mpimg.imread(rf'D:\KCL\year1\code\MTRobotSimulator_v2\v3\RobotConfigs\EndEffector.png')
base_image = mpimg.imread(rf'D:\KCL\year1\code\MTRobotSimulator_v2\v3\RobotConfigs\Base.png')

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class RobotArmSimulator:
    def __init__(self, master, robot: BasicRobotArm):
        self.master = master
        self.robot = robot
        self.arm_animation = False

        self.init_x0, self.init_y0 = 0.2111, 0.3111 # 0.2, 0.5

        #setting 1
        # self.max_x_vel, self.max_y_vel = 0.3, 0.3 # m/s
        # self.max_x_force, self.max_y_force = 0.8, 0.8
        # self.max_x_force, self.max_y_force = 0.4, 0.4

        self.max_x_vel, self.max_y_vel = 0.5, 0.5
        self.max_x_force, self.max_y_force = 0.5, 0.5  # 九个月的setting
        self.max_x_force, self.max_y_force = 1.2, 1.2
                
        self.positions = []
        self.velocities = []
        self.accelerations = []
        self.joint_angles = []
        self.joint_velocities = []
        self.joint_accelerations = []

        self.fig = Figure(figsize=(6, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)

        self.fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

        # self.ax.set_xlim(self.robot.x_min - 0.1, self.robot.x_max + 0.5)
        # self.ax.set_ylim(self.robot.y_min - 0.1, self.robot.y_max + 0.5)

        self.ax.set_xlim(-0.5, self.robot.x_max + 0.5)
        self.ax.set_ylim(-0.5, self.robot.y_max + 0.5)
        self.ax.set_aspect('equal', 'box')

        self.ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        self.ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
        self.ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
        self.ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))
        self.ax.grid(which='minor', color='gray', linestyle=':')
        self.ax.grid(which='major', color='gray', linestyle=':')
        
        self.ax.arrow(-0.3, 0, 2.5, 0, head_width=0.1, head_length=0.1, color='grey', overhang=0.2)  # 参数依次是起始点坐标、箭头的水平长度、箭头头部的宽度和长度，颜色等
        self.ax.arrow(0, -0.3, 0, 2.5, head_width=0.1, head_length=0.1, color='grey', overhang=0.2)

        self.starting_point, = self.ax.plot([], [], 'g*', ms=5)
        self.target_point, = self.ax.plot([], [], 'r*', ms=5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # initialize starting point
        self.robot.joint1_angle, self.robot.joint2_angle = self.robot.inverse_kinematics(self.init_x0, self.init_y0)
        x, y = self.robot.forward_kinematics_from_states_2_joints([self.robot.joint1_angle, self.robot.joint2_angle])

        if self.arm_animation == False:
            self.arm_line, = self.ax.plot([], [], 'o-', lw=3, alpha = 0.6) # robot arm toumingdu
            self.arm_line.set_data([0, x[0], x[1]], [0, y[0], y[1]])

        # self.arrow = self.ax.annotate('', xy=(x[1] + 2, y[1] + 1), xytext=(x[1],y[1]),arrowprops=dict(arrowstyle= '-|>',
        #                                                                                      color='blue', lw=0.5, ls='solid'))
        # self.force_arrow = self.ax.annotate('', xy=(x[1] + 2, y[1] + 1), xytext=(x[1],y[1]),arrowprops=dict(arrowstyle= '-|>',
        #                                                                                      color='red', lw=0.5, ls='solid'))


        self.arrow = self.ax.arrow(x[1],y[1],x[1] + 0.1, y[1] + 1, length_includes_head=True, head_width=0.05, head_length=0.1, fc='blue', ec='blue')
        self.force_arrow = self.ax.arrow(x[1],y[1],x[1] + 0.5, y[1] + 2, length_includes_head=True, head_width=0.05, head_length=0.1, fc='red', ec='red')
        
        self.draw_arm(ani=False)
        
    def reset_robot(self):
        self.robot.joint1_angle, self.robot.joint2_angle = self.robot.inverse_kinematics(self.init_x0, self.init_y0)
        
        self.robot.joint1_velocity = 0.0
        self.robot.joint2_velocity = 0.0

        self.robot.trajectory = []
        self.robot.velocity = []
        self.robot.joint1_angle_track = []
        self.robot.joint2_angle_track = []
        self.robot.joint1_velocity_track = []
        self.robot.joint2_velocity_track = []

    def draw_arm(self, vel_len_x=0, vel_len_y=0, force_len_x=0, force_len_y=0, ani=True, show_arm=True):
        x, y = self.robot.forward_kinematics_from_states_2_joints([self.robot.joint1_angle, self.robot.joint2_angle])

        if self.arm_animation == True:
            # Display/update the base image
            if hasattr(self, "base_image_display"):
                self.base_image_display.set_data(base_image)
                self.base_image_display.set_extent([-0.2, 0.2, -0.2, 0.2])  # Adjust these values for the desired size
            else:
                self.base_image_display = self.ax.imshow(base_image, extent=[-0.2, 0.2, -0.2, 0.2])

            # Display/update the joint images
            if hasattr(self, "joint1_image_display"):
                self.joint1_image_display.set_data(joint_image)
                self.joint1_image_display.set_extent([x[0]-0.1, x[0]+0.1, y[0]-0.1, y[0]+0.1])
            else:
                self.joint1_image_display = self.ax.imshow(joint_image, extent=[x[0]-0.1, x[0]+0.1, y[0]-0.1, y[0]+0.1])

            if hasattr(self, "joint2_image_display"):
                self.joint2_image_display.set_data(joint_image)
                self.joint2_image_display.set_extent([x[1]-0.1, x[1]+0.1, y[1]-0.1, y[1]+0.1])
            else:
                self.joint2_image_display = self.ax.imshow(joint_image, extent=[x[1]-0.1, x[1]+0.1, y[1]-0.1, y[1]+0.1])

            # Calculate the length and orientation for the first link
            length1 = np.sqrt(x[0]**2 + y[0]**2)
            angle1 = np.arctan2(y[0], x[0]) * (180/np.pi)

            # Calculate the scaling factor for the first link based on its length
            scale_factor1 = length1 / 1  # Assuming 0.6 is the default length of the link image

            # Create rotation and scaling transformations and display/update the link image for the first link
            rotation1 = Affine2D().rotate_deg(angle1).scale(scale_factor1, 1) + self.ax.transData
            if hasattr(self, "link1_image_display"):
                self.link1_image_display.set_data(link_image)
                self.link1_image_display.set_extent([0, length1, -0.1, 0.1])  # Adjust the y-values as needed
                self.link1_image_display.set_transform(rotation1)
            else:
                self.link1_image_display = self.ax.imshow(link_image, extent=[0, length1, -0.1, 0.1], transform=rotation1)

            # Similarly, calculate for the second link
            length2 = np.sqrt((x[1]-x[0])**2 + (y[1]-y[0])**2)
            angle2 = np.arctan2(y[1]-y[0], x[1]-x[0]) * (180/np.pi)

            # Calculate the scaling factor for the second link based on its length
            scale_factor2 = length2 / 1

            # Create rotation and scaling transformations and display/update the link image for the second link
            rotation2 = Affine2D().rotate_deg(angle2).translate(x[0], y[0]).scale(scale_factor2, 1) + self.ax.transData
            if hasattr(self, "link2_image_display"):
                self.link2_image_display.set_data(link_image)
                self.link2_image_display.set_extent([0, length2, -0.1, 0.1])  # Adjust the y-values as needed
                self.link2_image_display.set_transform(rotation2)
            else:
                self.link2_image_display = self.ax.imshow(link_image, extent=[0, length2, -0.1, 0.1], transform=rotation2)

        elif self.arm_animation == False:
            self.arm_line.set_data([0, x[0], x[1]], [0, y[0], y[1]])
            if not show_arm:
                self.arm_line.set_data([], [])

        self.draw_arrow(vel_len_x, vel_len_y, force_len_x, force_len_y)
        self.fig.canvas.draw()
        if ani:
            self.fig.canvas.flush_events()

    def draw_arrow(self, vel_len_x, vel_len_y, force_len_x, force_len_y):
        x, y = self.robot.forward_kinematics_from_states_2_joints([self.robot.joint1_angle, self.robot.joint2_angle])

        self.arrow.remove()
        self.force_arrow.remove()
        
        start_x = x[1]
        start_y = y[1]

        # end_x = x[1] + vel_len_x * 1.5 # change arrow length
        # end_y = y[1] + vel_len_y * 1.5
        # force_end_x = x[1] + force_len_x * 0.3 * 0.8 # change arrow length
        # force_end_y = y[1] + force_len_y * 0.3 * 0.8
    
        # velocity arrow
        # end_x = x[1] + vel_len_x * 1.0 # change arrow length
        # end_y = y[1] + vel_len_y * 1.0
        # force_end_x = x[1] + force_len_x * 1.0 # change arrow length
        # force_end_y = y[1] + force_len_y * 1.0

        # end_x = x[1] + vel_len_x * 0.1
        # end_y = y[1] + vel_len_y * 0.1
        # force_end_x = x[1] + force_len_x * 0.1
        # force_end_y = y[1] + force_len_y * 0.1

        # self.arrow = self.ax.annotate('', xy=(end_x, end_y), xytext=(start_x,start_y), arrowprops=dict(arrowstyle= '-|>',
        #                                                                                      color='blue', lw=0.1, ls='solid'))
        # self.force_arrow = self.ax.annotate('', xy=(force_end_x, force_end_y), xytext=(start_x,start_y), arrowprops=dict(arrowstyle= '-|>',
        #                                                                                      color='red', lw=1.0, ls='solid'))
        
        vel_end_x = vel_len_x
        vel_end_y = vel_len_y
        force_end_x = force_len_x
        force_end_y = force_len_y
        self.arrow = self.ax.arrow(start_x,start_y, vel_end_x, vel_end_y, length_includes_head=True, head_width=0.05, head_length=0.1, fc='blue', ec='blue')
        self.force_arrow = self.ax.arrow(start_x,start_y, force_end_x, force_end_y, length_includes_head=True, head_width=0.05, head_length=0.1, fc='red', ec='red')

    def learn(self, phi, force, demo=5):
        lam = 1e-4
        I = np.identity(demo)

        learned = (np.linalg.inv(phi @ phi.T + lam * I) @ phi @ (force.T)).T
        return learned
    
    def learn_yange(self, phi, force, demo=5):
        lam = 1e-6
        I = np.identity(demo)

        learned = (np.linalg.inv(phi @ phi.T + lam * I) @ phi @ (force.T)).T
        return learned

class TeacherRobot:
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
        # self.robot.set_noise_level(noise_mean, noise_std) 

        self.all_demo_comb = 0
        self.points_num = 0

    def reset(self, x0, y0, xd, yd):
        self.x0 = x0
        self.y0 = y0
        self.xd = xd
        self.yd = yd
        
        self.robot.reset()

    def teacher_demonstration(self, plot_animation_flag, update_step, L_kd):
        self.robot.joint1_angle, self.robot.joint2_angle = self.robot.inverse_kinematics(self.x0, self.y0)        
        L = np.array(L_kd)
        self.robot.set_skill_theta(L)
        #self.set_L(L=np.array([[-1, 0, -1, 0, (-1)*(-xd), 0], [0, -1, 0, -1, 0, (-1)*(-yd)]]))

        for i in range(update_step):
            self.robot.trajectory.append(list(self.robot.get_end_effector_position()))
            self.robot.velocity.append(list(self.robot.get_end_effector_velocity()))
            if plot_animation_flag:
                # Plot the current state of the robot arm and the trajectory
                plt.clf()
                plt.xlim(self.robot.x_min, self.robot.x_max)
                plt.ylim(self.robot.y_min, self.robot.y_max)
                plt.plot(self.x0, self.y0, marker="o", markersize=3, markeredgecolor="green", markerfacecolor="green", label='Starting Point')
                plt.plot(self.xd, self.yd, marker="o", markersize=3, markeredgecolor="red", markerfacecolor="red", label='Target Point')
                plt.legend(loc='upper right')
                plt.plot([0, self.robot.link1_length * np.cos(self.robot.joint1_angle), self.robot.link1_length * np.cos(self.robot.joint1_angle) + self.robot.link2_length * np.cos(self.robot.joint1_angle + self.robot.joint2_angle)], 
                            [0, self.robot.link1_length * np.sin(self.robot.joint1_angle), self.robot.link1_length * np.sin(self.robot.joint1_angle) + self.robot.link2_length * np.sin(self.robot.joint1_angle + self.robot.joint2_angle)],
                            '-o', color = 'blue')
                plt.plot([pos[0] for pos in self.robot.trajectory], [pos[1] for pos in self.robot.trajectory], '-r')
                plt.grid()
                plt.gca().set_aspect("equal")
                plt.draw()
                plt.title('Teacher: Frame {}'.format(i+1))
                plt.pause(0.001) 
            
            # robot.update() 
            self.robot.update_rk4() # updates using RK4 method

        self.trajectory = self.robot.trajectory
        self.velocity = self.robot.velocity

    def plot_dynamic_motion(self):
        # Create a figure with 4 subplots
        trajectory = np.array(self.robot.trajectory)
        velocity = np.array(self.robot.velocity)

        fig, axs = plt.subplots(4, 2)
        fig.suptitle("Skill 1's Dynamics Motion")
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
  
class RobotArmApp(tk.Tk):
    def __init__(self, robot: BasicRobotArm, demo_num, trial_num, pilot_num, force_teach, plot_result_figure_flag, std, show_guidace_flag, student_animation):
        super().__init__()
        self.geometry("1150x650") # width x height
        self.robot = robot

        self.slider_scale = 10000
        # self.target_point = [1.237, 0.592]
        self.target_point = [1.22857732, 0.58413413]
        self.L_real = np.zeros((2, 5))
        self.L_learner = np.zeros((2, 5))
 
        self.theta_1 = np.array([[-1, 0, -1, 0, 0.8],
                                [0, -1, 0, -1, 1.2]])   
        self.theta_2 = np.array([[0.5, 0, 0, 0, 0],
                    [0, 0.5, 0, 0, 0]])
        self.step = 0        

        # the data need to be recorded
        self.phi_recording = []
        self.force_recording = []
        self.teacher_trajectory = []

        self.user_number = -1
        self.create_page_1()
        self.clock_show_flag = 1
        
        self.demo_cnt= 0                # 5 demos in one trail
        self.max_demo_num = demo_num        # the number of demos
        # self.max_demo_num = 6        # the number of demos
        self.trail_cnt = 1              # 3 trials in one phase
        self.max_trail_num = trial_num  # 3 trails in each phase, input
        self.phase_cnt =  1             # 6 phase in total
        self.max_phase_num = 5

        self.force_teach_flag = force_teach
        self.plot_result_figure_flag = plot_result_figure_flag
        self.std_noise = std
        self.guidancetype = show_guidace_flag

        self.student_animation_show_flag = student_animation

        # data save initialization
        self.records = {
            'datas': [],
            'count': 0
        }
        

        self.arbitrary_state_1 = np.array([[ 0.12045094,  0.87406391,  1.39526239,  0.7190158,   1.33353343],
                                            [ 0.7308567,   0.42076512,  0.3578526,   1.34127574,  0.72742154],
                                            [ 0.23877311, -0.03683909,  -0.2930243,   0.04211806, -0.17467395],
                                            [ 0.04021354,  0.31865,    0.402504, -0.20321429, 0.15334464],
                                            [ 1.,          1.,          1.,          1.,          1.        ]])
        self.meaningful_state = np.array([[ 0.6,  0.6,  0.8,  0.8,  1.0],
                             [ 1.2,  1.2,  1.0,  1.0,  1.4],
                             [ 0.1,  0.5,  0,    0,    0.2],
                             [ 0,    0,    0.1,  0.5,  -0.4],
                             [ 1.,   1.,   1.,   1.,   1. ]])
        self.arbitrary_state_2 = np.array([[ 1.27147054,  1.32174012,  0.2311083,   1.41429467,  0.61247677],
                                          [ 1.49190387,  0.58612505,  1.11481334,  1.41065673,  1.33545924],
                                          [-0.19719273, -0.24874564,  0.44138709, -0.38701271, -0.07849007],
                                          [-0.38754479,  0.20116862,  0.09056881, -0.24202937, -0.25130862],
                                          [ 1.,          1.,          1.,          1.,          1.        ]])
        self.arbitrary_state_2 = np.array([[ 0.87147054,  1.32174012,  0.2311083,   1.41429467,  0.61247677],
                                            [ 1.59190387,  0.58612505,  1.11481334,  1.41065673,  1.33545924],
                                            [-0.19719273, -0.24874564,  0.44138709, -0.38701271, -0.07849007],
                                            [-0.38754479,  0.20116862,  0.09056881, -0.24202937, -0.25130862],
                                            [ 1.,          1.,          1.,          1.,          1.        ]])
        self.clever_state = np.array([[ 0.2,  0.8,  0.5,  1.2,  1.2],
                                      [ 1.2,  1.4,  1.0,  1.4,  0.6],
                                      [ 0.2,  0.,   0.1,  -0.1,  -0.2],
                                      [ 0.,   0.3,  0.1,  -0.4,  0.2],
                                      [ 1.,   1.,   1.,   1.,   1. ]])
        

        self.test_arbitrary_state = np.array([[ 0.12045094,  0.87406391,  1.39526239,  0.7190158,   1.33353343],
                                            [ 0.7308567,   0.42076512,  0.3578526,   1.34127574,  0.72742154],
                                            [ 0.23877311, -0.03683909,  -0.2930243,   0.04211806, -0.17467395],
                                            [ 0.04021354,  0.31865,    0.402504, -0.20321429, 0.15334464],
                                            [ 1.,          1.,          1.,          1.,          1.        ]])
        self.test_meaningful_state = np.array([[ 0.2,  0.8,  0.5,  1.2,  1.2],
                                      [ 1.2,  1.4,  1.0,  1.4,  0.6],
                                      [ 0.2,  0.,   0.1,  -0.1,  -0.2],
                                      [ 0.,   0.3,  0.1,  -0.4,  0.2],
                                      [ 1.,   1.,   1.,   1.,   1. ]])
        
        self.teaching_arbitrary_state = np.array([[ 0.87147054,  1.32174012,  0.2311083,   1.41429467,  0.61247677],
                                            [ 1.59190387,  0.58612505,  1.11481334,  1.41065673,  1.33545924],
                                            [-0.19719273, -0.24874564,  0.44138709, -0.38701271, -0.07849007],
                                            [-0.38754479,  0.20116862,  0.09056881, -0.24202937, -0.25130862],
                                            [ 1.,          1.,          1.,          1.,          1.        ]])
        # self.teaching_meaningful_state = np.array([[ 0.2,   0.84,  0.91,  0.82,  0.72],
        #                                             [ 0.2,   0.69,  0.98,  1.27,  1.44],
        #                                             [ 0.,    0.22, -0.03, -0.12, -0.],
        #                                             [ 0.,    0.35,  0.36,  0.23, -0.],
        #                                             [ 1.,    1.,    1.,    1.,    1.]])
        

        self.teaching_meaningful_state = np.array( [[ 1.92,  1.71,  0.61,  0.56,  0.72],
                                                    [ 0.36,  0.66,  1.27,  1.37,  1.34],
                                                    [ -0.3,  -0.62, -0.19,  0.06,  0.11],
                                                    [ 0.66,  0.62,  0.13,  0.05, -0.07],
                                                    [ 1.,    1.,    1.,    1.,    1.  ]])
        

        self.teaching_meaningful_state = np.array( [[ 2.,    1.92,  1.71,  0.61,  0.56  ],
                                                    [ 0.,    0.36,  0.66,  1.27,  1.37  ],
                                                    [ 0.,   -0.3,  -0.62, -0.19,  0.06  ],
                                                    [ 0.,    0.66,  0.62,  0.13,  0.05  ],
                                                    [ 1.,    1.,    1.,    1.,    1.    ]])
        
        # [[ 2.    1.92  1.71  0.61  0.56  0.72  0.8 ]
        # [ 0.    0.36  0.66  1.27  1.37  1.34  1.2 ]
        # [ 0.   -0.3  -0.62 -0.19  0.06  0.11  0.  ]
        # [ 0.    0.66  0.62  0.13  0.05 -0.07 -0.01]
        # [ 1.    1.    1.    1.    1.    1.    1.  ]]

        
        self.ordering = np.array([[0, 1, 2, 3, 4],
                                [2, 0, 1, 4, 3],
                                [0, 2, 3, 1, 4],
                                [3, 0, 1, 2, 4],
                                [0, 4, 2, 1, 3],
                                [4, 1, 3, 0, 2],
                                [1, 4, 2, 0, 3],
                                [3, 1, 4, 0, 2]])
        
        self.phase_1_state = self.test_arbitrary_state
        self.phase_4_state = self.test_arbitrary_state

        self.phase_2_state = self.test_meaningful_state
        self.phase_5_state = self.test_meaningful_state

        # if pilot_num == 2:
        #     self.phase_3_state = self.teaching_arbitrary_state
        # elif pilot_num == 3:
        #     self.phase_3_state = self.teaching_meaningful_state
        self.phase_3_state = self.teaching_meaningful_state


        # initial 
        self.given_state = self.phase_1_state
        
        self.teacher1_value(robot)
        # self.state_check_before_exp(self.arbitrary_state_1, self.arbitrary_state_2, self.meaningful_state, self.clever_state)
        self.state_check_before_exp(self.phase_1_state, self.phase_2_state, self.phase_3_state, self.phase_4_state, self.phase_5_state)
        
    def state_check_before_exp(self, state1, state2, state3, state4, state5):
        force1 = self.theta_1 @ state1
        force2 = self.theta_1 @ state2
        force3 = self.theta_1 @ state3
        force4 = self.theta_1 @ state4
        force5 = self.theta_1 @ state5

        if np.min(force1) < -0.5 or np.max(force1) > 0.5:
            print("state 1's force out of range.")
        if np.min(force2) < -0.5 or np.max(force2) > 0.5:
            print("state 2's force out of range.")
        if np.min(force3) < -0.5 or np.max(force3) > 0.5:
            print("state 3's force out of range.")
        if np.min(force4) < -0.5 or np.max(force4) > 0.5:
            print("state 4's force out of range.")
        if np.min(force5) < -0.5 or np.max(force5) > 0.5:
            print("state 5's force out of range.")


        force1_min, force1_max = np.amin(force1), np.amax(force1)
        force2_min, force2_max = np.amin(force2), np.amax(force2)
        force3_min, force3_max = np.amin(force3), np.amax(force3)
        force4_min, force4_max = np.amin(force4), np.amax(force4)
        force5_min, force5_max = np.amin(force5), np.amax(force5)

        # plot
        theta = np.linspace(0, np.pi / 2, 100)  # 创建100个点
        radius = 2
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        legend_labels = []

        sns.set_style('darkgrid')
        fig, axs = plt.subplots(1, 5, figsize=(18, 4), sharex=True)
        
        axs[0].scatter(state1[0,:], state1[1,:], color='red', marker='o', label='fixed state')  # 'o'代表圆点，color指定颜色
        axs[1].scatter(state2[0,:], state2[1,:], color='red', marker='o', label='fixed state')  # 'o'代表圆点，color指定颜色
        axs[2].scatter(state3[0,:], state3[1,:], color='red', marker='o', label='fixed state')  # 'o'代表圆点，color指定颜色
        axs[3].scatter(state4[0,:], state4[1,:], color='red', marker='o', label='fixed state')  # 'o'代表圆点，color指定颜色
        axs[4].scatter(state5[0,:], state5[1,:], color='red', marker='o', label='fixed state')
        for i in range(5):
            axs[0].arrow(state1[0,i], state1[1,i], state1[2,i], state1[3,i], head_width=0.1, head_length=0.1, color='blue', label='x velocity', overhang=0.6)
            axs[1].arrow(state2[0,i], state2[1,i], state2[2,i], state2[3,i], head_width=0.1, head_length=0.1, color='blue', label='x velocity', overhang=0.6)
            axs[2].arrow(state3[0,i], state3[1,i], state3[2,i], state3[3,i], head_width=0.1, head_length=0.1, color='blue', label='x velocity', overhang=0.6)
            axs[3].arrow(state4[0,i], state4[1,i], state4[2,i], state4[3,i], head_width=0.1, head_length=0.1, color='blue', label='x velocity', overhang=0.6)
            axs[4].arrow(state5[0,i], state5[1,i], state5[2,i], state5[3,i], head_width=0.1, head_length=0.1, color='blue', label='x velocity', overhang=0.6)
           
        labels = ['1', '2', '3', '4', '5']
        for label, xi, yi in zip(labels, state1[0,:], state1[1,:]):
            axs[0].annotate(label, (xi, yi), textcoords="offset points", xytext=(0,10), ha='center')
        for label, xi, yi in zip(labels, state2[0,:], state2[1,:]):
            axs[1].annotate(label, (xi, yi), textcoords="offset points", xytext=(0,10), ha='center')
        for label, xi, yi in zip(labels, state3[0,:], state3[1,:]):
            axs[2].annotate(label, (xi, yi), textcoords="offset points", xytext=(0,10), ha='center')
        for label, xi, yi in zip(labels, state4[0,:], state4[1,:]):
            axs[3].annotate(label, (xi, yi), textcoords="offset points", xytext=(0,10), ha='center')
        for label, xi, yi in zip(labels, state5[0,:], state5[1,:]):
            axs[4].annotate(label, (xi, yi), textcoords="offset points", xytext=(0,10), ha='center')
            
        for i in range(5):
            axs[i].plot(x, y, label='work space')
            axs[i].scatter(0.8, 1.2, marker='*', color='black')        
            axs[i].arrow(-0.3, 0, 2.5, 0, head_width=0.1, head_length=0.1, color='black', overhang=0.2)  # 参数依次是起始点坐标、箭头的水平长度、箭头头部的宽度和长度，颜色等
            axs[i].arrow(0, -0.3, 0, 2.5, head_width=0.1, head_length=0.1, color='black', overhang=0.2)

        axs[0].set_title('phase1 : pre-test arbitrary state')
        axs[1].set_title('phase2 : pre-test meaningful state')
        axs[2].set_title('phase3 : teaching state')
        axs[3].set_title('phase4 : later-test arbitrary state')
        axs[4].set_title('phase5 : later-test meaningful state')

        plt.figtext(0.2, 0.01, f'phase 1 force: ({force1_min:.3f}, {force1_max:.3f})', ha='center', va='bottom', fontsize=12)
        plt.figtext(0.35, 0.01, f'phase 2 force: ({force2_min:.3f}, {force2_max:.3f})', ha='center', va='bottom', fontsize=12)
        plt.figtext(0.5, 0.01, f'phase 3 force: ({force3_min:.3f}, {force3_max:.3f})', ha='center', va='bottom', fontsize=12)
        plt.figtext(0.65, 0.01, f'phase 4 force: ({force4_min:.3f}, {force4_max:.3f})', ha='center', va='bottom', fontsize=12)
        plt.figtext(0.8, 0.01, f'phase 5 force: ({force4_min:.3f}, {force4_max:.3f})', ha='center', va='bottom', fontsize=12)

        plt.show()


    def teacher1_value(self, robot):
        self.teacher1 = TeacherRobot(robot,0.2, 0.3, 0.8, 1.2)
        self.teacher1.teacher_demonstration(False, update_step=200, L_kd=self.theta_1)
        # teacher1.plot_dynamic_motion()

        self.teacher_trajectory = self.teacher1.trajectory
        self.teacher_velocity = self.teacher1.velocity
        self.teacher_q1_position = self.teacher1.robot.joint1_angle_track
        self.teacher_q2_position = self.teacher1.robot.joint1_angle_track
        self.teacher_q1_velocity = self.teacher1.robot.joint1_velocity_track
        self.teacher_q2_velocity = self.teacher1.robot.joint1_velocity_track


    def reset_canvas_config(self):        
        self.simulator.reset_robot()
        # self.simulator.draw_arm(ani=False)
        self.set_given_state()

    def reset_teaching_config(self):
        self.demo_cnt = 0

        self.phi_recording = []
        self.force_recording = []
    
    #--------------------------------------------------------------------------------#
    # page 1

    def create_page_1(self):
        self.page_1_frame = tk.Frame(self)
        self.page_1_frame.grid(row=0, column=0)

        self.create_canvas()
        self.create_intro()
        self.create_user_info_input()

        self.skill1_demo_btn.config(state='disabled')
        self.skill2_demo_btn.config(state='disabled')
 
    def create_canvas(self):
        self.plot_frame = tk.Frame(self)
        self.plot_frame.grid(row=0, column=2, rowspan=4, padx=20, pady=5)
        self.simulator = RobotArmSimulator(self.plot_frame, self.robot)
        self.simulator.draw_arm(ani=False)

    def create_intro(self):
        self.intro_frame = tk.Frame(self)
        self.intro_frame.grid(row=0, column=0, padx = 104, pady = 20)

        self.skill1_demo_btn = tk.Button(self.intro_frame, text="Skill 1 Movement", height=2, width=33, font=("Arial", 12), command=self.skill1_demo_callback)
        self.skill1_demo_btn.grid(row=0, column=0, sticky="nsew", pady = 20)

        self.skill2_demo_btn = tk.Button(self.intro_frame, text="Skill 2 Movement", height=2, width=33, font=("Arial", 12), command=self.skill2_demo_callback)
        self.skill2_demo_btn.grid(row=1, column=0, sticky="nsew") #pady = 5)

    def create_user_info_input(self):
        self.user_info_frame = tk.Frame(self)
        self.user_info_frame.grid(row=2, column=0) #, pady=140, sticky=tk.EW)

        user_entry_label = tk.Label(self.user_info_frame, text='Participant Number : ')
        user_entry_label.grid(row=0, column=0)

        self.user_entry = tk.Entry(self.user_info_frame)
        self.user_entry.grid(row=0, column=1, padx = 10)

        login_btn = tk.Button(self.user_info_frame, text="Log In", command=self.save_participant_number)
        login_btn.grid(row=0, column=2)

        self.title("User Info Page - Two-Axis RR Robot Arm Simulator")

    def save_participant_number(self):
        self.slide_frame = tk.Frame(self)
        value = self.user_entry.get()
        try:
            # self.user_number = int(value)
            self.user_number = value
            self.user_info_frame.destroy()
        except ValueError:
            messagebox.showwarning('Warning','Please input a valid participant ID.')
            self.quit()
        self.create_exp_start_btn()
        
    def create_exp_start_btn(self):
        self.exp_start_frame = tk.Frame(self)
        self.exp_start_frame.grid(row=2, column=0)

        start_btn = tk.Button(self.exp_start_frame, text="Start Experiment", height=2, width=33, font=("Arial", 12), command=self.exp_start_callback)
        start_btn.grid(row=0, column=0, sticky="nsew", pady = 20)

        self.title("Page 1 - Two-Axis RR Robot Arm Simulator")

    def exp_start_callback(self):
        self.exp_start_frame.destroy()
        self.create_countdown_clock()

        self.skill1_demo_btn.config(state='normal')
        self.skill2_demo_btn.config(state='normal')

    def create_countdown_clock(self):
        self.countdown_frame = tk.Frame(self)
        self.countdown_frame.grid(row=2, column=0) #, padx=50, pady=50)

        self.countdown_label = tk.Label(self.countdown_frame, font=("Arial", 30), bg="light grey")
        self.countdown_label.pack()

        # target_time = datetime.datetime.now() + datetime.timedelta(minutes=1) # 1 min counterdown clock gaishijian
        target_time = datetime.datetime.now() + datetime.timedelta(seconds=0) # 10 seconds counterdown clock

        self.clock_show_flag = 1
        self.update_countdown_clock(target_time)

    def update_countdown_clock(self, target_time):
        current_time = datetime.datetime.now()
        remaining_time = target_time - current_time

        if remaining_time.total_seconds() <= 0:
            self.clock_show_flag = 0
            self.countdown_label.config(text="Countdown Finished!", font=("Arial", 15))
            # Enable any buttons or perform any actions you want when the countdown finishes
            if self.skill1_demo_btn.cget("state") == 'normal' and self.skill2_demo_btn.cget("state") == 'normal':
                self.create_page_2()
                return

        minutes, seconds = divmod(remaining_time.seconds, 60)
        countdown_text = f"{minutes:02}:{seconds:02}"

        if self.clock_show_flag == 1:
            self.countdown_label.config(text=countdown_text)

        self.after(1000, self.update_countdown_clock, target_time)

    #--------------------------------------------------------------------------------#
    # page 2

    def create_page_2(self):
        self.title("Page 2 - Two-Axis RR Robot Arm Simulator")
        # self.reset_canvas_config()
        if self.user_number!= -1:
            self.page_1_frame.destroy()
            self.intro_frame.destroy()
            self.countdown_frame.destroy()
            
            self.create_phase_info_display()
            self.create_slider()
            self.create_button()
            # self.create_visual_guidance()

            self.reset_canvas_config()
            self.phase_selection(phase_num=self.phase_cnt)
            self.demonstrate_button.config(state='normal')
        else:
            messagebox.showerror('Please input a participant ID.')

    def create_phase_info_display(self):
        phase_frame = tk.Frame(self)
        phase_frame.grid(row=0, column=0, columnspan = 2)

        self.phase_display = tk.Label(phase_frame, text=f"Phase {self.phase_cnt}: Please teach Skill {(self.phase_cnt+1)%2 + 1} \n trial {self.trail_cnt} - demo {self.demo_cnt + 1}", 
                                      font=("Arial", 20))
        self.phase_display.grid(row=0, column=1, sticky="nsew")
        print("Phase {} - trial{} - demo{}".format(self.phase_cnt, self.trail_cnt, self.demo_cnt))
    

    def create_slider(self):
        self.slide_frame = tk.Frame(self)
        # self.slide_frame.grid(row=1, column=0, columnspan = 2)
        self.slide_frame.grid(row=1, column=0)
        if self.force_teach_flag:
            # x force slider label
            force_slider_label_x = tk.Label(self.slide_frame,text='Force, x min:', fg="red")
            force_slider_label_x.grid(row=0, column=0, sticky='e')
            self.force_slider_x = tk.Scale(self.slide_frame, from_=-self.simulator.max_x_force * self.slider_scale, to=self.simulator.max_x_force * self.slider_scale, 
                                            orient='horizontal', command=self.x_force_slider_changed, state='disabled', 
                                            length= 300, showvalue=0)
            self.force_slider_x.grid(row=0, column=1, pady=10)
            max3 = tk.Label(self.slide_frame,text='max', fg="red")
            max3.grid(row=0, column=2, sticky='w')
        
            # y force slider label
            force_slider_label_y = tk.Label(self.slide_frame,text='Force, y: min', fg="red")
            force_slider_label_y.grid(row=1, column=0, sticky='e')
            self.force_slider_y = tk.Scale(self.slide_frame, from_=-self.simulator.max_y_force * self.slider_scale, to=self.simulator.max_y_force * self.slider_scale, 
                                            orient='horizontal', 
                                            command=self.y_force_slider_changed, state='disabled', 
                                            length= 300, showvalue=0)
            self.force_slider_y.grid(row=1, column=1, pady=10)
            max6 = tk.Label(self.slide_frame,text='max', fg="red")
            max6.grid(row=1, column=2, sticky='w')

        self.set_given_state()
    
    
    def create_slider2(self):
        self.slide_frame = tk.Frame(self)
        self.slide_frame.grid(row=1, column=0, columnspan = 2)

        # x position slider label
        pos_slider_label_x = tk.Label(self.slide_frame,text='Position, x: min', fg="green")
        pos_slider_label_x.grid(row=0, column=0, sticky='e')
        self.position_slider_x = tk.Scale(self.slide_frame, from_= 0*self.slider_scale, to=self.simulator.robot.x_max*self.slider_scale, orient='horizontal', 
                                        state='normal', length= 300, showvalue=0)
        self.position_slider_x.grid(row=0, column=1, pady=0)
        
        # xdot velocity slider label
        vel_slider_label_x = tk.Label(self.slide_frame,text='Velocity, x: min', fg="blue")
        vel_slider_label_x.grid(row=1, column=0, sticky='e')
        self.velocity_slider_x = tk.Scale(self.slide_frame, from_=-self.simulator.max_x_vel*self.slider_scale, to=self.simulator.max_x_vel*self.slider_scale, orient='horizontal', 
                                          state='normal', length= 300, showvalue=0)
        self.velocity_slider_x.grid(row=1, column=1, pady=0)
        

        # y position slider label
        pos_slider_label_y = tk.Label(self.slide_frame,text='Position, y: min', fg="green")
        pos_slider_label_y.grid(row=4, column=0, sticky='e')
        self.position_slider_y = tk.Scale(self.slide_frame, from_=0*self.slider_scale, to=self.simulator.robot.y_max*self.slider_scale, orient='horizontal', 
                                          state='normal', length= 300, showvalue=0)
        self.position_slider_y.grid(row=4, column=1, pady=0)
        # ydot velocity slider label
        vel_slider_label_y = tk.Label(self.slide_frame,text='Velocity, y: min', fg="blue")
        vel_slider_label_y.grid(row=5, column=0, sticky='e')
        self.velocity_slider_y = tk.Scale(self.slide_frame, from_=-self.simulator.max_y_vel*self.slider_scale, to=self.simulator.max_y_vel*self.slider_scale, orient='horizontal', 
                                          state='normal', length= 300, showvalue=0)
        self.velocity_slider_y.grid(row=5, column=1, pady=0)
        
        gap_label = tk.Label(self.slide_frame,text='            ', fg="green")
        gap_label.grid(row=3, column=0)


        if self.force_teach_flag:
            # x force slider label
            force_slider_label_x = tk.Label(self.slide_frame,text='Force, x min:', fg="red")
            force_slider_label_x.grid(row=2, column=0, sticky='e')
            
            self.force_slider_x = tk.Scale(self.slide_frame, from_=-self.simulator.max_x_force * self.slider_scale, to=self.simulator.max_x_force * self.slider_scale, 
                                            orient='horizontal', 
                                            command=self.x_force_slider_changed, state='disabled', 
                                            length= 300, showvalue=0)
            self.force_slider_x.grid(row=2, column=1, pady=0)
        
            # y force slider label
            force_slider_label_y = tk.Label(self.slide_frame,text='Force, y: min', fg="red")
            force_slider_label_y.grid(row=6, column=0, sticky='e')
            # self.force_slider_y = tk.Scale(self.slide_frame, from_=-self.simulator.max_y_force*10000, to=self.simulator.max_y_force*10000, orient='horizontal', 
            #                                 command=self.y_force_slider_changed, state='disabled', 
            #                                 length= 300, showvalue=0)

            self.force_slider_y = tk.Scale(self.slide_frame, from_=-self.simulator.max_y_force * self.slider_scale, to=self.simulator.max_y_force * self.slider_scale, 
                                            orient='horizontal', 
                                            command=self.y_force_slider_changed, state='disabled', 
                                            length= 300, showvalue=0)
            self.force_slider_y.grid(row=6, column=1, pady=0)

        max1 = tk.Label(self.slide_frame,text='max', fg="green")
        max1.grid(row=0, column=2, sticky='w')
        max2 = tk.Label(self.slide_frame,text='max', fg="green")
        max2.grid(row=1, column=2, sticky='w')
        max3 = tk.Label(self.slide_frame,text='max', fg="green")
        max3.grid(row=2, column=2, sticky='w')
        max4 = tk.Label(self.slide_frame,text='max', fg="green")
        max4.grid(row=4, column=2, sticky='w')
        max5 = tk.Label(self.slide_frame,text='max', fg="green")
        max5.grid(row=5, column=2, sticky='w')
        max6 = tk.Label(self.slide_frame,text='max', fg="green")
        max6.grid(row=6, column=2, sticky='w')

        self.set_given_state()
    
    def create_button(self):
        button_frame = tk.Frame(self)
        button_frame.grid(row=2, column=0) #, pady=5, sticky=tk.EW)

        self.demonstrate_button = tk.Button(button_frame, text="Record", height=2, width=11, font=("Arial", 12), command=self.demonstrate_btn_callback)
        self.demonstrate_button.grid(row=0, column=0, padx = 10)
        self.demonstrate_button.config(state='disabled')

        self.next_demo_button = tk.Button(button_frame, text="Next Demo", height=2, width=11, font=("Arial", 12), command=self.next_demo_btn_callback)
        self.next_demo_button.grid(row=0, column=1, padx = 10)
        self.next_demo_button.config(state='disabled')

        self.next_button = tk.Button(button_frame, text="Next Trial", height=2, width=11, font=("Arial", 12), command=self.next_phase_btn_callback)
        self.next_button.grid(row=0, column=2, padx = 10)
        self.next_button.config(state='disabled') # 初始化next button就是disabled的
    

    def create_score_display(self, guidance):
        self.score_frame = tk.Frame(self)
        self.score_frame.grid(row=3, column=0)
        if guidance:
            # score_label = tk.Label(self.score_frame,text='Score is :')
            # score_label.grid(row=0, column=0)
            # self.score_text = tk.Text(self.score_frame, width=30, height=1, wrap=tk.WORD)
            # self.score_text.grid(row=0, column=1)  # Adjusted pady
            # self.score_text.insert(tk.END, '\n          ')
            # self.score_text.see(tk.END)

            guidance_label = tk.Label(self.score_frame,text='Guidance is :')
            guidance_label.grid(row=1, column=0)

            self.guidance_text = tk.Text(self.score_frame, width=40, height=15, wrap=tk.WORD)
            self.guidance_text.grid(row=1, column=1)  # Adjusted pady
        else:
            no_guidance_label1 = tk.Label(self.score_frame,width=30, height=17, text='No Guidance')
            no_guidance_label1.grid(row=1, column=0)
            no_guidance_label2 = tk.Label(self.score_frame,width=25, height=17, text='')
            no_guidance_label2.grid(row=1, column=1)

    def create_visual_guidance(self):
        # self.visual_guidance_frame = tk.Frame(self)
        # # self.slide_frame.grid(row=1, column=0, columnspan = 2)
        # self.visual_guidance_frame.grid(row=0, column=3)

        self.fig, self.ax = plt.subplots(subplot_kw={"projection": "3d"})
        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.get_tk_widget().grid(row=0, column=3, columnspan=3)

        # final_heights = [float(entry.get()) for entry in self.final_heights_entries]
        final_heights = final_heights = [1.5, 0.8, -1.5, 1.2, 0.7]
        self.draw_3d_plot(final_heights)
        self.canvas.draw()

    # 定义高斯突起函数
    def gaussian_bump(self, x, y, x0, y0, sigma):
        return np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    
    def draw_5_3d_plot(self, final_heights):
        self.ax.cla()

        # 创建一个平面网格
        x = np.linspace(0, 2, 100)
        y = np.linspace(-1.0, 1.0, 100)
        x, y = np.meshgrid(x, y)

        # 定义五个突起的位置、参数和目标最终高度
        positions = [(1.0, 1.0), (1.5, 1.0), (0.5, 1.0), (1.0, 0.5), (1.5, 0.75)]
        sigmas = [0.1, 0.1, 0.1, 0.1, 0.1]  # 每个突起的宽度（标准差）

        # 应用五个高斯突起函数，并调整以达到指定的最终高度
        z = -x - y + 1.0
        for (x0, y0), sigma, final_height in zip(positions, sigmas, final_heights):
            # 计算原始平面在突起位置的高度
            base_height = -x0 - y0 + 1.0
            # 计算突起相对于原始平面的高度增加值
            relative_height = final_height - base_height
            # 仅在高斯函数的基础上增加高度
            z += self.gaussian_bump(x, y, x0, y0, sigma) * relative_height

        # 绘制图形
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        # ax.plot_surface(x, y, z, cmap='viridis', alpha=0.6)  # 半透明平面
        self.ax.plot_surface(x, y, z, cmap='viridis', alpha=0.6)

        # z = -2 的投影平面
        z_projection_level = -2
        z_plane = np.full_like(x, z_projection_level)  # 创建一个 z = -2 的平面
        # ax.plot_surface(x, y, z_plane, color='grey', alpha=0.3)  # 灰色透明平面
        self.ax.plot_surface(x, y, z_plane, color='grey', alpha=0.3)

        # 添加垂直线、标记点和文本标签
        for i, ((x0, y0), final_height) in enumerate(zip(positions, final_heights), start=1):
            ax.plot([x0, x0], [y0, y0], [final_height, z_projection_level], color='red', linestyle='--')
            ax.scatter([x0], [y0], [z_projection_level], color='green')
            ax.scatter(x0, y0, final_height, color='red', label=f'{i}th force')
            # 添加文本标签
            ax.text(x0, y0, final_height, f'{i}', color='blue')

        # 设置坐标轴标签        
        self.ax.set_xlabel('X position', fontsize=15)
        self.ax.set_ylabel('X velocity', fontsize=15)
        self.ax.set_zlabel('X force', fontsize=15)
        self.fig.legend(loc=1)
        
    def draw_3d_plot(self, final_heights):
        self.ax.cla()

        # 创建一个平面网格
        x = np.linspace(0, 2, 100)
        y = np.linspace(-1.0, 1.0, 100)
        x, y = np.meshgrid(x, y)

        # 定义五个突起的位置、参数和目标最终高度
        positions = [(1.0, 1.0)]
        sigmas = [0.1]  # 每个突起的宽度（标准差）

        # 应用五个高斯突起函数，并调整以达到指定的最终高度
        z = -x - y + 1.0
        for (x0, y0), sigma, final_height in zip(positions, sigmas, final_heights):
            # 计算原始平面在突起位置的高度
            base_height = -x0 - y0 + 1.0
            # 计算突起相对于原始平面的高度增加值
            relative_height = final_height - base_height
            # 仅在高斯函数的基础上增加高度
            z += self.gaussian_bump(x, y, x0, y0, sigma) * relative_height

        # 绘制图形
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        self.ax.plot_surface(x, y, z, cmap='viridis', alpha=0.6)

        # z = -2 的投影平面
        z_projection_level = -2
        z_plane = np.full_like(x, z_projection_level)  # 创建一个 z = -2 的平面
        # ax.plot_surface(x, y, z_plane, color='grey', alpha=0.3)  # 灰色透明平面
        self.ax.plot_surface(x, y, z_plane, color='grey', alpha=0.3)

        # 添加垂直线、标记点和文本标签
        for i, ((x0, y0), final_height) in enumerate(zip(positions, final_heights), start=1):
            ax.plot([x0, x0], [y0, y0], [final_height, z_projection_level], color='red', linestyle='--')
            ax.scatter([x0], [y0], [z_projection_level], color='green')
            ax.scatter(x0, y0, final_height, color='red', label=f'{i}th force')
            # 添加文本标签
            ax.text(x0, y0, final_height, f'{i}', color='blue')

        # 设置坐标轴标签        
        self.ax.set_xlabel('X position', fontsize=15)
        self.ax.set_ylabel('X velocity', fontsize=15)
        self.ax.set_zlabel('X force', fontsize=15)
        self.fig.legend(loc=1)

    def update_plot(self):
        # print("Xxxxxxxxxxxxxxxxxxxxxxxxxxx", self.force_recording)
        # final_heights = self.force_recording[-1][0]
        # print("Xxxxxxxxxxxxxxxxxxxxxxxxxxx", final_heights)
        
        # self.draw_3d_plot(final_heights)
        # self.canvas.draw()
        pass



    def L_set(self, skill_num):
        # skill 1: converge; skill 2: oscillation
        if skill_num == 1:
            self.target_point = [0.8, 1.2]
            self.step = 200
            self.L_real = self.theta_1
            self.simulator.reset_robot()
        elif skill_num == 2:
            # self.target_point = [1.237, 0.592] 
            self.target_point = [1.22857732, 0.58413413]
            self.step = 200
            self.L_real = self.theta_2
            self.simulator.reset_robot()
        else:
            raise ValueError('Invalid Skill Number.')

    def show_guidance_for_each_demonstration(self, guidance):
        self.guidance_text.insert(tk.END, f"{self.demo_cnt+1} demonstration completed.\n")
        self.guidance_text.insert(tk.END, f"Guidance: ({'+' if -guidance[0] >= 0 else ''}{-guidance[0]:.3f}, {'+' if -guidance[1] >= 0 else ''}{-guidance[1]:.3f})\n")
        self.guidance_text.see(tk.END)

    def demonstrate_btn_callback(self):
        if self.demo_cnt <= self.max_demo_num-1:
            self.demonstrate_button.config(state="disabled")
            x = self.simulator.robot.get_end_effector_position()[0]
            y = self.simulator.robot.get_end_effector_position()[1]
            self.phi_recording.append([self.given_state[0, self.demo_cnt], self.given_state[1, self.demo_cnt], self.given_state[2, self.demo_cnt], self.given_state[3, self.demo_cnt]])
           
            if self.force_teach_flag:
                self.force_recording.append([self.force_slider_x.get()/self.slider_scale, self.force_slider_y.get()/self.slider_scale])
            
            print(f"{self.demo_cnt+1} demonstration completed.")
            # print("phi input:   ", self.phi_recording[self.demo_cnt] + [1])
            # print("force input: ", self.force_recording[self.demo_cnt])
            # print("real force:  ", self.L_real @ np.array(self.phi_recording[self.demo_cnt] + [1]))
            # print("force error :  ", self.force_recording[self.demo_cnt] - self.L_real @ np.array(self.phi_recording[self.demo_cnt] + [1]))
            
            # guidance2 = self.guidance_calculator(self.given_state, self.force_recording[self.demo_cnt])
            guidance2 = self.force_recording[self.demo_cnt] - self.L_real @ np.array(self.phi_recording[self.demo_cnt] + [1])
            
            print(f"Real theta {self.L_real[0]}, {self.L_real[1]}")
            print("Each state", self.phi_recording[self.demo_cnt] + [1])
            print("Force Input: ", self.force_recording[self.demo_cnt])
            print("Force Real", self.L_real @ np.array(self.phi_recording[self.demo_cnt] + [1]))
            print("Force error / Guidance :", guidance2)
            print("\n")

            # self.log_action("Button clicked")
            self.log_action(f'{self.phase_cnt} - {self.trail_cnt} - {self.demo_cnt}')
            self.log_action(f'{self.phi_recording[self.demo_cnt]}')
            self.log_action(f'{self.force_recording[self.demo_cnt]}')
            self.log_action(f'{self.L_real @ np.array(self.phi_recording[self.demo_cnt] + [1])}')
            self.log_action(f'GUIDANCE: {guidance2}')
        

            if self.phase_cnt == 3:
                self.show_guidance_for_each_demonstration(guidance2)
            self.demo_cnt += 1    
            
            self.demonstrate_button.config(state="disabled")
            self.next_demo_button.config(state="normal")
            self.lock_slider()
            self.update_plot()

            if self.demo_cnt == self.max_demo_num:
                self.lock_slider()
                messagebox.showinfo("Information", "This is what we learned.")
                self.demonstrate_button.config(state='disabled')
                self.next_demo_button.config(state="disabled")
                self.next_button.config(state='normal')
                # self.teach_button.config(state='normal')

                self.teaching()
                self.lock_slider()
                return
            


    def next_demo_btn_callback(self):
        self.unlock_slider()
        self.set_given_state()
        self.reset_slider()
        
        self.phase_display.config(text=f"Phase {self.phase_cnt}: Please teach Converge Skill \n trial {self.trail_cnt} - demo {self.demo_cnt + 1}")
        
        self.demonstrate_button.config(state="disabled")
        self.next_demo_button.config(state="disabled")
    
    def next_phase_btn_callback(self):
        self.unlock_slider()
        self.reset_slider()

        self.demonstrate_button.config(state='normal')
        self.next_demo_button.config(state="disabled")
        self.next_button.config(state='disabled')
        
        if self.trail_cnt < self.max_trail_num: # trail 1, trial 2 , only in phase 3
            new_order = self.ordering[self.trail_cnt]
            self.given_state = self.phase_3_state[:, new_order]
            print('trail:', new_order)
            self.trail_cnt += 1
        else:                                   # reach max trails, change phase
            if self.phase_cnt <= self.max_phase_num:
                self.phase_cnt += 1
            self.trail_cnt = 1                  # reset trail = 1

        self.phase_display.config(text=f"Phase {self.phase_cnt}: Please teach Converge Skill \n trial {self.trail_cnt} - demo {self.demo_cnt + 1}")
        # if self.phase_cnt == 3 and self.demo_cnt > 1:
        #     self.guidance_text.insert(tk.END, f"-----------------------------------\n")  # 显示分数，小数点后四位
        #     self.guidance_text.see(tk.END)

        # phase change -> change skill 
        if self.phase_cnt in [1, 2, 3, 4, 5] and self.trail_cnt == 1:
            self.phase_selection(phase_num=self.phase_cnt)
            print("reset all skill's parameter, phase and trail :", self.phase_cnt, self.trail_cnt)
    
        print("Phase {} - {} - {}".format(self.phase_cnt, self.trail_cnt, self.demo_cnt))
        self.reset_canvas_config()
    
    def phase_selection(self, phase_num):
        self.reset_teaching_config()
        self.unlock_slider()

        # set robot
        # self.simulator.starting_point.set_data(0.2, 0.3)
        self.simulator.starting_point.set_data([], []) # dont show starting point, 因为这是从任意一个点。
        self.simulator.target_point.set_data(0.8, 1.2)
        self.L_set(1)
        self.teacher_trajectory = self.teacher1.trajectory
    
        self.simulator.robot.set_skill_theta(self.L_real)
        self.set_given_state()
        self.simulator.draw_arm(ani=False, show_arm=False)

        if phase_num == 1:
            self.max_trail_num = 1
            self.given_state = self.phase_1_state
            self.student_animation_show_flag = True
        elif phase_num == 2:
            self.max_trail_num = 1
            self.given_state = self.phase_2_state
            self.student_animation_show_flag = True
        elif phase_num == 3:
            self.create_score_display(True)
            self.max_trail_num = 8
            print("xxxxxxxxxxxxxxxxxxxxxx", self.trail_cnt)
            new_order = self.ordering[self.trail_cnt-1]
            self.given_state = self.phase_3_state[:, new_order]
            print('phase: ', new_order)
            self.student_animation_show_flag = True
        elif phase_num == 4:
            self.create_score_display(False)
            self.max_trail_num = 1
            self.given_state = self.phase_4_state
            self.student_animation_show_flag = False
        elif phase_num == 5:
            self.create_score_display(False)
            self.max_trail_num = 1
            self.given_state = self.phase_5_state
            self.student_animation_show_flag = False
        
    def set_given_state(self):
        # self.position_slider_x.config(state='normal')
        # self.position_slider_y.config(state='normal')
        # self.velocity_slider_x.config(state='normal')
        # self.velocity_slider_y.config(state='normal')
        if self.demo_cnt <= self.max_demo_num:
            x_position = self.given_state[0, self.demo_cnt]
            y_position = self.given_state[1, self.demo_cnt]
            x_velocity = self.given_state[2, self.demo_cnt]
            y_velocity = self.given_state[3, self.demo_cnt] 
        
            joint_angles = self.simulator.robot.inverse_kinematics(x_position, y_position)
            self.simulator.robot.joint1_angle, self.simulator.robot.joint2_angle = joint_angles[0], joint_angles[1]
            # self.simulator.draw_arm(x_velocity, y_velocity, self.force_slider_x.get() / 10000, self.force_slider_y.get() / 10000, ani=False)
            self.simulator.draw_arm(x_velocity, y_velocity, 0,0)

            # self.position_slider_x.set(x_position * self.slider_scale)
            # self.position_slider_y.set(y_position * self.slider_scale)
            # self.velocity_slider_x.set(x_velocity * self.slider_scale)
            # self.velocity_slider_y.set(y_velocity * self.slider_scale)

            # self.position_slider_x.config(state='disabled')
            # self.position_slider_y.config(state='disabled')
            # self.velocity_slider_x.config(state='disabled')
            # self.velocity_slider_y.config(state='disabled')

    def teaching(self):
        if len(self.phi_recording) == 0:
            messagebox.showwarning('Warning', 'No demonstration has recorded.')
            return
        elif len(self.phi_recording) !=0 and len(self.phi_recording) != 5:
            messagebox.showwarning('Warning', 'Incomplete demonstration.')
            return

        # phi = np.vstack((np.array(self.phi_recording).T, np.ones((1, 5))))
        phi = self.given_state

        if self.force_teach_flag == False: # force from calculated with noise
            force_real = self.L_real @ phi
            force_noi_diff = self.force_std_noise * np.random.randn(2, 5)
            force_with_noise = force_real + force_noi_diff
            force_given = force_with_noise
        elif self.force_teach_flag == True:
            force_teached = np.array(self.force_recording).T # np.shape (5,2)
            force_given = force_teached
        
        # print("phiphiphiphiphiphiphi", phi)
        # print("phiphiphiphiphiphiphi", force_given)

        # calculate theta
        self.L_learner = self.simulator.learn(phi, force_given)

        print(f"Phase: {self.phase_cnt} - {self.trail_cnt}: self.L_real = {self.L_real[0]}, {self.L_real[1]}")
        print(f"Phase: {self.phase_cnt} - {self.trail_cnt}: self.L_learner = {self.L_learner[0]}, {self.L_learner[1]}")

        self.simulator.robot.set_skill_theta(self.L_learner)

        # show student's animation
        self.next_button.config(state='disabled')
        if self.phase_cnt in [1, 2, 3, 4, 5]:
            self.simulator.robot.joint1_angle, self.simulator.robot.joint2_angle = self.simulator.robot.inverse_kinematics(0.2, 0.3)
            
        self.simulator.robot.joint1_velocity, self.simulator.robot.joint2_velocity = 0.0, 0.0
        self.simulator.starting_point.set_data(0.2, 0.3) # plot starting point
        for i in range(self.step):
            self.simulator.robot.trajectory.append(list(self.simulator.robot.get_end_effector_position()))
            self.simulator.robot.velocity.append(list(self.simulator.robot.get_end_effector_velocity()))
            if self.student_animation_show_flag:
                self.simulator.draw_arm()
            self.simulator.robot.update_rk4()
        self.student_trajectory = self.simulator.robot.trajectory
        self.simulator.starting_point.set_data([], [])

        # calculate error
        rmse = np.sqrt(np.mean((np.array(self.teacher_trajectory) - np.array(self.student_trajectory))**2))
        nmse1, nmse2, el2 = self.error_calculator(self.L_real, self.L_learner)
        print(f"rmse: {rmse}, nmse1: {nmse1}, nmse2: {nmse2}, el2: {el2}")

        # 为了探究不同的lamba的影响，记录数据
        L_learner_yange = self.simulator.learn_yange(phi, force_given)
        nmse1_yange, nmse2_yange, el2_yange = self.error_calculator(self.L_real, L_learner_yange)


        # calculate score, but only show on 3th phase
        guidance_score = self.score_calculator(phi, force_given)
        if self.phase_cnt == 3 and self.guidancetype:
        #     self.score_text.insert(tk.END, '\n {}.'.format(guidance_score)) # display score
        #     self.score_text.see(tk.END)
            self.guidance_text.insert(tk.END, f"-----------------------------------\n")  # 显示分数，小数点后四位
            self.guidance_text.see(tk.END)

        # store data tocsv file
        self.store_data_tocsv(self.phase_cnt, self.trail_cnt, phi, force_given, self.L_learner, rmse, nmse1, nmse2, el2, guidance_score, L_learner_yange, nmse1_yange, nmse2_yange, el2_yange)
        self.store_data_tojson(phi, force_given, self.L_learner, rmse, nmse1, nmse2, el2, guidance_score, L_learner_yange, nmse1_yange, nmse2_yange, el2_yange)
        
        if self.plot_result_figure_flag:
            self.plot_user_result()
        
        # reset robot
        self.simulator.reset_robot()
        self.reset_teaching_config()
        self.next_button.config(state='normal')

        # when phase_cnt = max phases, exp. finished.
        if self.phase_cnt >= self.max_phase_num and self.trail_cnt == self.max_trail_num:
            print("Finished!")
            messagebox.showinfo("Information", "Experiment Finished.")
            self.quit()

    def x_force_slider_changed(self, value):
        # print("x force slider value:", value)
        self.demonstrate_button.config(state = "normal")
        self.simulator.draw_arm(self.given_state[2, self.demo_cnt], self.given_state[3, self.demo_cnt], self.force_slider_x.get() / self.slider_scale, force_len_y=self.force_slider_y.get() / self.slider_scale, ani=False)
        
    def y_force_slider_changed(self, value):
        # print("y force slider value:", value)
        self.demonstrate_button.config(state = "normal")
        self.simulator.draw_arm(self.given_state[2, self.demo_cnt], self.given_state[3, self.demo_cnt], self.force_slider_x.get() / self.slider_scale, force_len_y=self.force_slider_y.get() / self.slider_scale, ani=False)
    
    def score_calculator(self, phi, force):
        pass
    
    def guidance_calculator(self, phi, force):
        real_force = self.L_real @ self.given_state
        force_error = real_force - force
        return force_error
    
    def error_calculator(self, theta_real, theta_learned):
        nmse1, nmse2 = self.nmse_cal(theta_real, theta_learned)
        el2 = self.el2_cal(theta_real, theta_learned)
        return nmse1, nmse2, el2
    
    def nmse_cal(self, theta_real, theta_learned):
        test_phi = np.array([[1.54547433, 1.82535471, 0.20575465, 0.40244344, 0.09569714, 1.04273663, 1.72299245, 0.95102198, 0.68314063, 0.61546897],
                            [0.08959529, 0.01713778, 1.64400987, 0.53837856, 0.17259851, 1.47867964, 0.31134094, 0.59224916, 0.6820265, 1.03555592],
                            [-0.12833334, -0.10372383, -0.00663499, 0.1608878, -0.10024024, -0.26926053, -0.40315198, 0.23943124, -0.1348775, 0.05278681],
                            [0.15063608, 0.39135257, 0.16674541, -0.07797693, 0.38231418, 0.25929959, -0.25023862, 0.44598129, -0.46304699, -0.47250088],
                            [1,1,1,1,1,1,1,1,1,1]])
        n = test_phi.shape[1]
        
        force_desired = np.dot(theta_real, test_phi)
        force_learned = np.dot(theta_learned, test_phi)

        nmse1 = np.mean((force_learned[0, :]-force_desired[0, :])**2) / np.var(force_desired[0, :], ddof=0)
        nmse2 = np.mean((force_learned[1, :]-force_desired[1, :])**2) / np.var(force_desired[1, :], ddof=0)
        
        print("nmse:", nmse1, nmse2)
        return nmse1, nmse2

    def el2_cal(self, theta_real, theta_learned):
        el2 = np.linalg.norm(theta_real - theta_learned)
        return el2

    def store_data_tocsv(self, phase, trail, phi, force, theta_learned, rmse, nmse1, nmse2, el2, score, L_learner_yange, nmse1_yange, nmse2_yange, el2_yange):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        target_dir = os.path.join(os.path.join(current_dir, f'formal_exp_results'), f"formal_exp_id_{self.user_number}")
        os.makedirs(target_dir, exist_ok=True)
        filename = os.path.join(target_dir, f"formal_exp_id_{self.user_number}.csv")
        
        # store value
        with open(filename, mode="a", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)

            csvwriter.writerows([["phase", f"{phase}", "trail",f"{trail}"]])
            csvwriter.writerow([])

            # Write phi matrix
            csvwriter.writerow(["phi"])
            csvwriter.writerows(phi)
            csvwriter.writerow([])  # Empty row between matrices
            
            # Write force matrix
            csvwriter.writerow(["force"])
            csvwriter.writerows(force)

            csvwriter.writerow(["theta_learned"])
            csvwriter.writerows(theta_learned)
            csvwriter.writerow([])
            
            csvwriter.writerows([["rmse", f"{rmse}"]])
            csvwriter.writerows([["nmse1", f"{nmse1}"]])
            csvwriter.writerows([["nmse2", f"{nmse2}"]])
            csvwriter.writerows([["el2", f"{el2}"]])
            csvwriter.writerow([])

            csvwriter.writerows([["score", f"{score}"]])
            csvwriter.writerow([])

            if self.phase_cnt == 3:
                csvwriter.writerows([["score", f"{score}"]])
                csvwriter.writerow([])

            csvwriter.writerow(["theta_learned yange"])
            csvwriter.writerows(L_learner_yange)
            csvwriter.writerow([])
            
            csvwriter.writerows([["nmse1 yange", f"{nmse1_yange}"]])
            csvwriter.writerows([["nmse2 yange", f"{nmse2_yange}"]])
            csvwriter.writerows([["el2 yange", f"{el2_yange}"]])
            csvwriter.writerow([])

        print("Matrix saved to", filename)
        print("-----------------------------------------")  
    
    def store_data_tojson(self, phi, force_given, L, rmse, nmse1, nmse2, el2, score, L_learner_yange, nmse1_yange, nmse2_yange, el2_yange):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        target_dir = os.path.join(os.path.join(current_dir, f'formal_exp_results'), f"formal_exp_id_{self.user_number}")
        filename = os.path.join(target_dir, f"formal_exp_id_{self.user_number}.json")

        if os.path.exists(filename):
            with open(filename, 'r') as f:
                self.records = json.load(f)

        self.records['datas'].append({
            'phi': phi,
            'force': force_given,
            'theta': L,
            'rmse': rmse,
            'nmse1': nmse1,
            'nmse2': nmse2,
            'el2': el2, 
            'score': score,
            'theta_yange': L_learner_yange, 
            'nmse1_yange': nmse1_yange, 
            'nmse2_yange': nmse2_yange, 
            'el2_yange': el2_yange
        })
        self.records['count'] += 1
        
        with open(filename, 'w') as f:
            json.dump(self.records, f, cls=NumpyEncoder)
    
    
        
    def skill1_demo_callback(self):
        self.skill1_demo_btn.config(state='disabled')
        self.skill2_demo_btn.config(state='disabled')
        self.L_set(1)

        self.simulator.target_point.set_data([0.8], [1.2])

        starting_pts = [[0.3, 1.3], [0.7, 0.1], [1.6, 1.3], [0.1, 0.5]]
        for num in range(len(starting_pts)+1):
            self.simulator.init_x0, self.simulator.init_y0 = starting_pts[num-1]
            self.simulator.starting_point.set_data([self.simulator.init_x0], [self.simulator.init_y0])
            self.simulator.reset_robot()
            self.simulator.robot.set_skill_theta(self.L_real)

            x, y = self.simulator.robot.get_end_effector_position()
            while (x-self.target_point[0])**2 + (y-self.target_point[1])**2 > 0.0005:
                self.simulator.robot.trajectory.append(list(self.simulator.robot.get_end_effector_position()))
                self.simulator.robot.velocity.append(list(self.simulator.robot.get_end_effector_velocity()))

                self.simulator.draw_arm()
                
                self.simulator.robot.update_rk4()

                x, y = self.simulator.robot.get_end_effector_position()
        
        # self.simulator.init_x0, self.simulator.init_y0 = 0.2, 0.5
        self.simulator.init_x0, self.simulator.init_y0 = 0.2111, 0.3111
        self.simulator.starting_point.set_data([], [])
        self.simulator.target_point.set_data([], [])
        self.simulator.reset_robot()

        self.skill1_demo_btn.config(state='normal')
        self.skill2_demo_btn.config(state='normal')
    
    def skill2_demo_callback(self):
        self.skill1_demo_btn.config(state='disabled')
        self.skill2_demo_btn.config(state='disabled')
        self.simulator.init_x0, self.simulator.init_y0 = 0.8, 1.1
        self.simulator.reset_robot()
        self.L_set(2)
        self.simulator.robot.set_skill_theta(self.L_real)

        self.simulator.starting_point.set_data(0.8, 1.1)
        # self.simulator.target_point.set_data(1.237, 0.592)
        self.simulator.target_point.set_data(1.22857732, 0.58413413)

        for i in range(self.step):
            self.simulator.robot.trajectory.append(list(self.simulator.robot.get_end_effector_position()))
            self.simulator.robot.velocity.append(list(self.simulator.robot.get_end_effector_velocity()))

            self.simulator.draw_arm()
            
            self.simulator.robot.update_rk4()

        self.simulator.starting_point.set_data([], [])
        self.simulator.target_point.set_data([], [])
        self.skill1_demo_btn.config(state='normal')
        self.skill2_demo_btn.config(state='normal')

        # self.simulator.init_x0, self.simulator.init_y0 = 0.2, 0.5
        self.simulator.init_x0, self.simulator.init_y0 = 0.2111, 0.3111
        self.simulator.reset_robot()

    def lock_slider(self):
        if self.force_teach_flag:
            self.force_slider_x.config(state="disabled") 
            self.force_slider_y.config(state="disabled") 

    def unlock_slider(self):
        if self.force_teach_flag:
            self.force_slider_x.config(state="normal") 
            self.force_slider_y.config(state="normal") 

    def reset_slider(self):
        if self.force_teach_flag:
                self.force_slider_x.set(0.0)
                self.force_slider_y.set(0.0)

    def log_action(self, action):
        with open("user_log.txt", "a") as log_file:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_file.write(f"{timestamp}: {action}\n")

    def plot_user_result(self):
        path = f'User{self.user_number}-Phase{self.phase_cnt}-Trial{self.trail_cnt}-Choices'
        self.plot_function(self.given_state, self.force_recording, path)

        path2 = f'User{self.user_number}-Phase{self.phase_cnt}-Trial{self.trail_cnt}-Dynamics'
        self.plot_dynamic_motion(path2)
        
    def plot_function(self, state, force, path):
        state_matrix = np.array(state)
        force = np.array(self.force_recording).T

        theta = np.linspace(0, np.pi / 2, 100)  # 创建100个点
        radius = 2
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        legend_labels = []

        # 绘制1/4圆弧
        plt.plot(x, y, label='work space')
        plt.scatter(0.8, 1.2, marker='*', color='black')

        plt.scatter(state_matrix[0,:], state_matrix[1,:], color='grey', marker='o', label='Fixed State\'s Position')  # 'o'代表圆点，color指定颜色
        plt.scatter(0.8, 1.2, marker='*', color='red', label = 'target point')

        plt.arrow(state_matrix[0,0], state_matrix[1,0], state_matrix[2,0], state_matrix[3,0], head_width=0.1, head_length=0.1, color='blue', label='velocity', overhang=0.6)
        plt.arrow(state_matrix[0,0], state_matrix[1,0], force[0,0], force[1,0], head_width=0.1, head_length=0.1, color='red', label='force', overhang=0.6)
        for i in range(1, 5):
            # plt.arrow(state_matrix[0,i], state_matrix[1,i], state_matrix[2,i], 0, head_width=0.1, head_length=0.1, color='blue', label='x velocity', overhang=0.6)
            # plt.arrow(state_matrix[0,i], state_matrix[1,i], 0, state_matrix[3,i], head_width=0.1, head_length=0.1, color='red', label='y velocity', overhang=0.6)
            plt.arrow(state_matrix[0,i], state_matrix[1,i], state_matrix[2,i], state_matrix[3,i], head_width=0.1, head_length=0.1, color='blue', overhang=0.6)
            plt.arrow(state_matrix[0,i], state_matrix[1,i], force[0,i], force[1,i], head_width=0.1, head_length=0.1, color='red', overhang=0.6)
            
        labels = ['1', '2', '3', '4', '5']
        for label, xi, yi in zip(labels, state_matrix[0,:], state_matrix[1,:]):
            plt.annotate(label, (xi, yi), textcoords="offset points", xytext=(0,10), ha='center')

        plt.arrow(-0.3, 0, 2.5, 0, head_width=0.1, head_length=0.1, color='black', overhang=0.2)  # 参数依次是起始点坐标、箭头的水平长度、箭头头部的宽度和长度，颜色等
        plt.arrow(0, -0.3, 0, 2.5, head_width=0.1, head_length=0.1, color='black', overhang=0.2)
        
        plt.title(f'Fixed State and Action sketch \nUser {self.user_number}-Phase {self.phase_cnt}- Trial {self.trail_cnt}-Choices')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend(loc=1)

        plt.grid(True)  # 添加网格线
        plt.axis('equal')  # 设置坐标轴比例相等
        plt.xlim(-0.3,2.5)
        plt.ylim(-0.3,2.5)
        
        save_path = f"D:\KCL\year1\code\MTRobotSimulator_v2\9monthReview\{path}.png"
        plt.savefig(save_path)
        plt.clf()

    def plot_dynamic_motion(self, path):
        # Create a figure with 4 subplots
        trajectory = np.array(self.simulator.robot.trajectory)
        velocity = np.array(self.simulator.robot.velocity)

        teacher_trajectory = np.array(self.teacher_trajectory)
        teacher_velocity = np.array(self.teacher_velocity)
        teacher_q1_position = np.array(self.teacher_q1_position)
        teacher_q2_position = np.array(self.teacher_q2_position)
        teacher_q1_velocity = np.array(self.teacher_q1_velocity)
        teacher_q2_velocity = np.array(self.teacher_q2_velocity)


        fig, axs = plt.subplots(4, 2, sharex=True)
        fig.suptitle(f'Dynamics Motion \n User {self.user_number}-Phase {self.phase_cnt}- Trial {self.trail_cnt}-Choices')

        idx = [i for i in range(len(self.simulator.robot.joint1_angle_track))]

        axs[0, 0].plot(idx, teacher_q1_position, 'red', label = 'desired behaviour', linestyle = ':')
        axs[0, 1].plot(idx, teacher_q2_position, 'red',  linestyle = ':')
        axs[1, 0].plot(idx, teacher_q1_velocity, 'red',  linestyle = ':')
        axs[1, 1].plot(idx, teacher_q2_velocity, 'red',  linestyle = ':')
        axs[2, 0].plot(idx, teacher_trajectory[:, 0], 'red',  linestyle = ':')
        axs[2, 1].plot(idx, teacher_trajectory[:, 1], 'red',  linestyle = ':')
        axs[3, 0].plot(idx, teacher_velocity[:, 0], 'red',  linestyle = ':')
        axs[3, 1].plot(idx, teacher_velocity[:, 1], 'red',  linestyle = ':')
        
        
        axs[0, 0].plot(idx, self.simulator.robot.joint1_angle_track, label = 'learner\'s behaviour', linewidth = 0.8)
        axs[0, 1].plot(idx, self.simulator.robot.joint2_angle_track, linewidth = 0.8)
        axs[1, 0].plot(idx, self.simulator.robot.joint1_velocity_track, linewidth = 0.8)
        axs[1, 1].plot(idx, self.simulator.robot.joint2_velocity_track, linewidth = 0.8)
        axs[2, 0].plot(idx, trajectory[:, 0], linewidth = 0.8)
        axs[2, 1].plot(idx, trajectory[:, 1], linewidth = 0.8)
        axs[3, 0].plot(idx, velocity[:, 0], linewidth = 0.8)
        axs[3, 1].plot(idx, velocity[:, 1], linewidth = 0.8)

        axs[0, 0].set_ylabel('q1')
        axs[0, 1].set_ylabel('q2')
        axs[1, 0].set_ylabel('q1_dot')
        axs[1, 1].set_ylabel('q2_dot')
        axs[2, 0].set_ylabel('x')
        axs[2, 1].set_ylabel('y')
        axs[3, 0].set_ylabel('x_dot')
        axs[3, 1].set_ylabel('y_dot')
        axs[3, 0].set_xlabel('times')
        axs[3, 1].set_xlabel('times')

        fig.legend(loc=1)
        fig.tight_layout()
        
        save_path = f"D:\KCL\year1\code\MTRobotSimulator_v2\9monthReview\{path}.png"
        plt.savefig(save_path)
        plt.clf()

        


if __name__ == "__main__":
    robot = BasicRobotArm(link1_length=1, link2_length=1, 
                        link1_mass=1, link2_mass=1, 
                        joint1_angle=0.0, joint2_angle=0.0, 
                        joint1_velocity=0.0, joint2_velocity=0.0, 
                        joint1_torque=0.0, joint2_torque=0.0,
                        time_step=0.05, g=9.81)
    
    x0, y0 = 0.2, 1.5
    xt, yt = 1.5, 0.3

    app = RobotArmApp(robot=robot, demo_num=5, trial_num=1, pilot_num=3, force_teach=True, plot_result_figure_flag=False, std=0.0, show_guidace_flag = True, student_animation = True)
    app.mainloop()