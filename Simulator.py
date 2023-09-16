import os
import csv
import math
import datetime
import numpy as np
import tkinter as tk

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import tkinter.messagebox as messagebox
from sklearn.preprocessing import normalize
import matplotlib.image as mpimg
from matplotlib.transforms import Affine2D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from BasicRobotArm import BasicRobotArm

joint_image = mpimg.imread("./RobotConfigs/Joint.png")
link_image = mpimg.imread("./RobotConfigs/Link.png")
# Inside the class initialization
end_effector_image = mpimg.imread("./RobotConfigs/EndEffector.png")
base_image = mpimg.imread("./RobotConfigs/Base.png")

class RobotArmSimulator:
    def __init__(self, master, robot: BasicRobotArm, starting_point, target_point):
        self.master = master
        self.robot = robot
        self.arm_animation = False

        self.init_x0, self.init_y0 = starting_point

        self.x0, self.y0 = starting_point
        self.xt, self.yt = target_point

        self.max_x_vel, self.max_y_vel = 1, 1 # m/s
        self.max_x_force, self.max_y_force = 3.5, 3.5 # N

        self.positions = []
        self.velocities = []
        self.accelerations = []
        self.joint_angles = []
        self.joint_velocities = []
        self.joint_accelerations = []

        self.fig = Figure(figsize=(6, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)

        self.ax.set_xlim(self.robot.x_min - 0.5, self.robot.x_max + 0.5)
        self.ax.set_ylim(self.robot.y_min - 0.5, self.robot.y_max + 0.5)
        self.ax.set_aspect('equal', 'box')
        self.ax.grid(True)

        self.starting_point, = self.ax.plot([], [], 'r*', ms=5)
        self.target_point, = self.ax.plot([], [], 'g*', ms=5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # initialize
        self.robot.joint1_angle, self.robot.joint2_angle = self.robot.inverse_kinematics(self.x0, self.y0)
        x, y = self.robot.forward_kinematics_from_states_2_joints([self.robot.joint1_angle, self.robot.joint2_angle])

        if self.arm_animation == False:
            self.arm_line, = self.ax.plot([], [], 'o-', lw=3)
            self.arm_line.set_data([0, x[0], x[1]], [0, y[0], y[1]])
        # display fen velocity
        # self.arrow_x = self.ax.annotate('', xy=(x[1] + 2, y[1]), xytext=(x[1],y[1]),arrowprops=dict(arrowstyle= '-|>',
        #                                                                                      color='blue', lw=0.5, ls='--'))
        # self.arrow_y = self.ax.annotate('', xy=(x[1], y[1]+1), xytext=(x[1], y[1]),arrowprops=dict(arrowstyle= '-|>',
        #                                                                                      color='blue', lw=0.5, ls='--'))
        self.arrow = self.ax.annotate('', xy=(x[1] + 2, y[1] + 1), xytext=(x[1],y[1]),arrowprops=dict(arrowstyle= '-|>',
                                                                                             color='blue', lw=0.5, ls='solid'))
        self.force_arrow = self.ax.annotate('', xy=(x[1] + 2, y[1] + 1), xytext=(x[1],y[1]),arrowprops=dict(arrowstyle= '-|>',
                                                                                             color='red', lw=0.5, ls='solid'))
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

    def reconfig_robot(self, starting_point, target_point):
        self.init_x0, self.init_y0 = starting_point

        self.x0, self.y0 = starting_point
        self.xt, self.yt = target_point

    def reset_robot_to_teach(self):
        self.robot.joint1_angle, self.robot.joint2_angle = self.robot.inverse_kinematics(self.init_x0, self.init_y0)

    def draw_arm(self, vel_len_x=0, vel_len_y=0, force_len_x=0, force_len_y=0, ani=True, show_arm=True, show_start_pt=True):
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
        self.starting_point.set_data([self.x0],[self.y0])
        self.target_point.set_data([self.xt], [self.yt])

        

        if not show_start_pt:
            self.starting_point.set_data([], [])

        # self.draw_arrow(vel_len_x, vel_len_y)
        self.draw_arrow(vel_len_x, vel_len_y, force_len_x, force_len_y)
        self.fig.canvas.draw()
        if ani:
            self.fig.canvas.flush_events()

    def draw_arrow(self, vel_len_x, vel_len_y, force_len_x, force_len_y):
        x, y = self.robot.forward_kinematics_from_states_2_joints([self.robot.joint1_angle, self.robot.joint2_angle])

        # display fen velocity
        # self.arrow_x.remove()
        # self.arrow_y.remove()
        # self.arrow_x = self.ax.annotate('', xy=(end_x, start_y), xytext=(start_x,start_y), arrowprops=dict(arrowstyle= '-|>',
        #                                                                                      color='grey', lw=0.1, ls='--'))
        # self.arrow_y = self.ax.annotate('', xy=(start_x, end_y), xytext=(start_x,start_y), arrowprops=dict(arrowstyle= '-|>',
        #                                                                                      color='grey', lw=0.1, ls='--'))
        
        self.arrow.remove()
        self.force_arrow.remove()
        
        start_x = x[1]
        start_y = y[1]
        end_x = x[1] + vel_len_x * 0.5 # change arrow length
        end_y = y[1] + vel_len_y * 0.5

        force_end_x = x[1] + force_len_x * 0.2 # change arrow length
        force_end_y = y[1] + force_len_y * 0.2

        self.arrow = self.ax.annotate('', xy=(end_x, end_y), xytext=(start_x,start_y), arrowprops=dict(arrowstyle= '-|>',
                                                                                             color='blue', lw=0.8, ls='solid'))
        
        self.force_arrow = self.ax.annotate('', xy=(force_end_x, force_end_y), xytext=(start_x,start_y), arrowprops=dict(arrowstyle= '-|>',
                                                                                             color='red', lw=1.0, ls='solid'))
        

    def set_x0y0(self, x0, y0):
        self.x0 = x0
        self.y0 = y0

    def set_xtyt(self, xt, yt):
        self.xt = xt
        self.yt = yt
    
    def learn(self, phi, force, demo=5):
        lam = 1e-7
        I = np.identity(demo)

        learned = (np.linalg.inv(phi @ phi.T + lam * I) @ phi @ (force.T)).T

        return learned
    
class RobotArmApp(tk.Tk):
    def __init__(self, robot: BasicRobotArm, demo_num, trial_num, force_teach, std):
        super().__init__()
        self.title("Two-Axis RR Robot Arm Simulator")
        self.geometry("1150x650") # width x height

        self.robot = robot
        self.starting_point = [0.2, 0.5] #[0, 0]
        self.target_point = [0, 0]
        self.target_point_plot = [100, 100]
        self.L = np.zeros((2, 5))
        self.L_learner = np.zeros((2, 5))
        self.step = 0        

        self.mode = ''

        # the data need to be recorded
        self.phi_recording = []
        self.force_recording = []
        self.teacher_trajectory = []

        self.participant_number = -1
        self.create_welcome_page()
        self.clock_show_flag = 1
        
        self.demo_cnt= 0                # 5 demos in one trail
        self.demo_num = demo_num        # the number of demos
        self.trail_cnt = 1              # 3 trials in one phase
        self.max_trail_num = trial_num  # 3 trails in each phase, input
        self.phase_cnt = 1              # 6 phase in total

        self.force_teach_flag = force_teach
        self.std_noise = std

        self.show_strating_point = True
        
    def reset_canvas_config(self):
        self.starting_point = [0.2, 0.5] #[0, 0]
        self.target_point = [0, 0]
        self.target_point_plot = [100, 100]
        self.L = np.zeros((2, 5))
        self.step = 0

        # self.simulator.reconfig_robot(starting_point=self.starting_point, target_point=self.target_point)
        self.simulator.reconfig_robot(starting_point=self.starting_point, target_point=self.target_point_plot)
        self.simulator.reset_robot()

        self.simulator.draw_arm(ani=False)

    def reset_teaching_config(self):
        self.mode = ''

        self.demo_cnt = 0

        self.phi_recording = []
        self.force_recording = []
        
        self.L_learner = np.zeros((2, 5))
        self.teacher_trajectory = []

    def create_welcome_page(self):
        self.welcome_page_frame = tk.Frame(self)
        self.welcome_page_frame.grid(row=0, column=0)

        self.create_canvas()
        self.create_play_mode()
        self.create_participant_number()

        self.skill1_demo_btn.config(state='disabled')
        self.skill2_demo_btn.config(state='disabled')

    def create_exp_start_btn(self):
        self.expstart_frame = tk.Frame(self)
        self.expstart_frame.grid(row=2, column=0) #, padx=50, pady=50)

        start_btn = tk.Button(self.expstart_frame, text="Start Experiment", height=2, width=33, font=("Arial", 12), command=self.exp_start_callback)
        start_btn.grid(row=0, column=0, sticky="nsew", pady = 20)

    def exp_start_callback(self):
        self.expstart_frame.destroy()
        self.create_countdown_clock()

        self.skill1_demo_btn.config(state='normal')
        self.skill2_demo_btn.config(state='normal')

    def create_countdown_clock(self):
        self.countdown_frame = tk.Frame(self)
        self.countdown_frame.grid(row=2, column=0) #, padx=50, pady=50)

        self.countdown_label = tk.Label(self.countdown_frame, font=("Arial", 30), bg="light grey")
        self.countdown_label.pack()

        # target_time = datetime.datetime.now() + datetime.timedelta(minutes=3) # 1 min counterdown clock gaishijian
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
                self.formal_teaching_callback()
                return

        minutes, seconds = divmod(remaining_time.seconds, 60)
        countdown_text = f"{minutes:02}:{seconds:02}"

        if self.clock_show_flag == 1:
            self.countdown_label.config(text=countdown_text)

        self.after(1000, self.update_countdown_clock, target_time)

    def create_canvas(self):
        self.plot_frame = tk.Frame(self)
        self.plot_frame.grid(row=0, column=2, rowspan=4, padx=20, pady=5)
        # self.simulator = RobotArmSimulator(self.plot_frame, self.robot, self.starting_point, self.target_point)
        self.simulator = RobotArmSimulator(self.plot_frame, self.robot, self.starting_point, self.target_point_plot)
        self.simulator.draw_arm(ani=False)

    def create_play_mode(self):
        self.play_mode_frame = tk.Frame(self)
        self.play_mode_frame.grid(row=0, column=0, padx = 104, pady = 20)
        # self.play_mode_frame.grid(row=0, column=0, columnspan=2, pady=5, sticky=tk.EW)

        self.skill1_demo_btn = tk.Button(self.play_mode_frame, text="Skill 1 Movement", height=2, width=33, font=("Arial", 12), command=self.skill1_demo_callback)
        self.skill1_demo_btn.grid(row=0, column=0, sticky="nsew", pady = 20)

        self.skill2_demo_btn = tk.Button(self.play_mode_frame, text="Skill 2 Movement", height=2, width=33, font=("Arial", 12), command=self.skill2_demo_callback)
        self.skill2_demo_btn.grid(row=1, column=0, sticky="nsew") #pady = 5)

    def L_set(self, skill_num):
        if skill_num == 1:
            self.starting_point = [0.8, 1.1] #[0.8, 1.1]
            self.target_point = [0.0, 0.0]
            self.target_point_plot = [1.23, 0.59]

            # self.simulator.reconfig_robot(starting_point=self.starting_point, target_point=self.target_point)
            self.simulator.reconfig_robot(starting_point=self.starting_point, target_point=self.target_point_plot)
            self.simulator.reset_robot()

            L_kd = np.array([[2, 0, 0, 0], [0, 2, 0, 0]])
            L_d = np.array([[L_kd[0,0] * (-self.target_point[0])], [L_kd[1,1] * (-self.target_point[1])]])
            self.L = np.concatenate((L_kd, L_d), axis=1)

            self.step = 75
            self.show_strating_point = True
        elif skill_num == 2:
            self.starting_point = [0.2, 0.3]
            self.target_point = [1.2, 1.2]

            self.simulator.reconfig_robot(starting_point=self.starting_point, target_point=self.target_point)
            self.simulator.reset_robot()

            L_kd = np.array([[-3, 0, -7, 0], [0, -3, 0, -7]])
            # L_d = np.array([[L_kd[0,0] * (-self.target_point[0]), 0], [0, L_kd[1,1] * (-self.target_point[1])]])
            L_d = np.array([[L_kd[0,0] * (-self.target_point[0])], [L_kd[1,1] * (-self.target_point[1])]])
            self.L = np.concatenate((L_kd, L_d), axis=1)

            self.step = 153
            self.show_strating_point = False
        else:
            raise ValueError('Invalid Skill Number.')

    def skill1_demo_callback(self):
        self.skill1_demo_btn.config(state='disabled')
        self.skill2_demo_btn.config(state='disabled')
        self.L_set(1)
        self.simulator.robot.set_L(self.L)
        if (self.target_point[0] - 0) ** 2 + (self.target_point[1] - 0) ** 2 <= 2 ** 2:
            for i in range(self.step):
                self.simulator.robot.trajectory.append(list(self.simulator.robot.get_end_effector_position()))
                self.simulator.robot.velocity.append(list(self.simulator.robot.get_end_effector_velocity()))

                self.simulator.draw_arm()
                
                self.simulator.robot.update_rk4()
        self.skill1_demo_btn.config(state='normal')
        self.skill2_demo_btn.config(state='normal')

    def skill2_demo_callback(self):
        self.skill1_demo_btn.config(state='disabled')
        self.skill2_demo_btn.config(state='disabled')
        self.L_set(2)

        starting_pts = [[0.3, 1.3], [0.7, 0.1], [1.6, 1.3], [0.1, 0.5]]
        for num in range(len(starting_pts)+1):
            self.simulator.reconfig_robot(starting_point=self.starting_point, target_point=self.target_point)
            self.simulator.reset_robot()
            self.simulator.robot.set_L(self.L)
            if (self.target_point[0] - 0) ** 2 + (self.target_point[1] - 0) ** 2 <= 2 ** 2:
                x, y = self.simulator.robot.get_end_effector_position()
                while (x-self.target_point[0])**2 + (y-self.target_point[1])**2 > 0.0005:
                    self.simulator.robot.trajectory.append(list(self.simulator.robot.get_end_effector_position()))
                    self.simulator.robot.velocity.append(list(self.simulator.robot.get_end_effector_velocity()))

                    self.simulator.draw_arm()
                    
                    self.simulator.robot.update_rk4()

                    x, y = self.simulator.robot.get_end_effector_position()
            if num < len(starting_pts):
                self.starting_point = starting_pts[num]

        self.skill1_demo_btn.config(state='normal')
        self.skill2_demo_btn.config(state='normal')
            
    def create_participant_number(self):
        self.participant_frame = tk.Frame(self)
        self.participant_frame.grid(row=2, column=0) #, pady=140, sticky=tk.EW)

        user_entry_label = tk.Label(self.participant_frame, text='Participant Number')
        user_entry_label.grid(row=0, column=0)

        self.user_entry = tk.Entry(self.participant_frame)
        self.user_entry.grid(row=0, column=1, padx = 10)

        login_btn = tk.Button(self.participant_frame, text="Log In", command=self.save_participant_number)
        login_btn.grid(row=0, column=2)

    def save_participant_number(self):
        value = self.user_entry.get()
        try:
            self.participant_number = int(value)
            self.participant_frame.destroy()
        except ValueError:
            # print("Error: Invalid input. Please enter an integer.")
            messagebox.showwarning('Warning','Please input a valid participant ID.')
            self.quit()
        self.create_exp_start_btn()
        
    def formal_teaching_callback(self):
        self.reset_canvas_config()
        if self.participant_number!= -1:
            self.welcome_page_frame.destroy()
            self.play_mode_frame.destroy()
            self.countdown_frame.destroy()

            self.create_slider()
            self.create_skill_mode_display()
            self.create_teach_demo_button()
            self.create_next_button()

            self.reset_canvas_config()
            self.mode_selection(skill_var=1)
        else:
            messagebox.showerror('Please input a valid participant ID.')
    
    def create_slider(self):
        self.slide_frame = tk.Frame(self)
        self.slide_frame.grid(row=1, column=0, columnspan = 2)

        # x position slider label
        pos_slider_label_x = tk.Label(self.slide_frame,text='Position, x:   min', fg="green")
        pos_slider_label_x.grid(row=0, column=0)
        self.position_slider_x = tk.Scale(self.slide_frame, from_= 0*10000, to=self.simulator.robot.x_max*10000, orient='horizontal', 
                                        command=self.x_pos_slider_changed, state='disabled', 
                                        length= 300, showvalue=0)
        self.position_slider_x.grid(row=0, column=1, pady =15)
        
        # x slider value
        self.pos_val_x = tk.StringVar()
        self.pos_val_x.set('max')
        self.pos_label_x = tk.Label(self.slide_frame, textvariable=self.pos_val_x, width=10, fg="green")
        self.pos_label_x.grid(row=0, column=2)

        # y position slider label
        pos_slider_label_y = tk.Label(self.slide_frame,text='Position, y:   min', fg="green")
        pos_slider_label_y.grid(row=1, column=0)
        self.position_slider_y = tk.Scale(self.slide_frame, from_=0*10000, to=self.simulator.robot.y_max*10000, orient='horizontal', 
                                          command=self.y_pos_slider_changed, state='disabled', 
                                          length= 300, showvalue=0)
        self.position_slider_y.grid(row=1, column=1, pady=15)

        # y slider value
        self.pos_val_y = tk.StringVar()
        self.pos_val_y.set('max')
        self.pos_label_y = tk.Label(self.slide_frame, textvariable=self.pos_val_y, width=10, fg="green")
        self.pos_label_y.grid(row=1, column=2)

        # xdot velocity slider label
        vel_slider_label_x = tk.Label(self.slide_frame,text='Velocity, x:   min', fg="blue")
        vel_slider_label_x.grid(row=2, column=0)
        self.velocity_slider_x = tk.Scale(self.slide_frame, from_=-self.simulator.max_x_vel*5000, to=self.simulator.max_x_vel*5000, orient='horizontal', 
                                          command=self.x_vel_slider_changed, state='disabled', 
                                          length= 300, showvalue=0)
        self.velocity_slider_x.grid(row=2, column=1, pady=15)
        # xdot slider value
        self.vel_val_x = tk.StringVar()
        self.vel_val_x.set("max")
        self.vel_label_x = tk.Label(self.slide_frame, textvariable=self.vel_val_x,  fg="blue")
        self.vel_label_x.grid(row=2, column=2)

        # ydot velocity slider label
        vel_slider_label_y = tk.Label(self.slide_frame,text='Velocity, y:   min', fg="blue")
        vel_slider_label_y.grid(row=3, column=0, padx=20)
        self.velocity_slider_y = tk.Scale(self.slide_frame, from_=-self.simulator.max_y_vel*5000, to=self.simulator.max_y_vel*5000, orient='horizontal', 
                                          command=self.y_vel_slider_changed, state='disabled', 
                                          length= 300, showvalue=0)
        self.velocity_slider_y.grid(row=3, column=1, pady=15)
        # ydot slider value
        self.vel_val_y = tk.StringVar()
        self.vel_val_y.set("max")
        self.vel_label_y = tk.Label(self.slide_frame, textvariable=self.vel_val_y, fg="blue")
        self.vel_label_y.grid(row=3, column=2)


        if self.force_teach_flag:
            # x force slider label
            force_slider_label_x = tk.Label(self.slide_frame,text='Force, x:   min', fg="red")
            force_slider_label_x.grid(row=4, column=0, padx=20)
            self.force_slider_x = tk.Scale(self.slide_frame, from_=-self.simulator.max_x_force*5000, to=self.simulator.max_x_force*5000, orient='horizontal', 
                                            command=self.x_force_slider_changed, state='disabled', 
                                            length= 300, showvalue=0)
            self.force_slider_x.grid(row=4, column=1, pady=15)
            # x force slider value
            self.force_val_x = tk.StringVar()
            self.force_val_x.set("max")
            self.force_val_x = tk.Label(self.slide_frame, textvariable=self.force_val_x, fg="red")
            self.force_val_x.grid(row=4, column=2)
            self.force_slider_value_x = tk.Label(self.slide_frame,text='None', fg="red")
            self.force_slider_value_x.grid(row=4, column=3)
            self.force_slider_value_x.grid_forget()

            # y force slider label
            force_slider_label_y = tk.Label(self.slide_frame,text='Force, y:   min', fg="red")
            force_slider_label_y.grid(row=5, column=0, padx=20)
            self.force_slider_y = tk.Scale(self.slide_frame, from_=-self.simulator.max_y_force*5000, to=self.simulator.max_y_force*5000, orient='horizontal', 
                                            command=self.y_force_slider_changed, state='disabled', 
                                            length= 300, showvalue=0)
            self.force_slider_y.grid(row=5, column=1, pady=15)
            # y force slider value
            self.force_val_y = tk.StringVar()
            self.force_val_y.set("max")
            self.force_val_y = tk.Label(self.slide_frame, textvariable=self.force_val_y, fg="red")
            self.force_val_y.grid(row=5, column=2)
            self.force_slider_value_y = tk.Label(self.slide_frame,text='None', fg="red")
            self.force_slider_value_y.grid(row=5, column=3)
            self.force_slider_value_y.grid_forget()
    
    def create_skill_mode_display(self):
        skill_frame = tk.Frame(self)
        skill_frame.grid(row=0, column=0, columnspan = 2)

        # self.text_display = tk.Label(skill_frame, text="Phase "+str(self.phase_cnt) +": Please teach Skill 1", font=("Arial", 20))
        # self.text_display = tk.Label(skill_frame, text="Please teach Skill 1 (Phase "+str(self.phase_cnt) , font=("Arial", 20))
        self.text_display = tk.Label(skill_frame, text="Phase 1: Please teach Skill 1", font=("Arial", 20))
        print("Phase {} - {}".format(self.phase_cnt, self.trail_cnt))
        self.text_display.grid(row=0, column=1, sticky="nsew")

        self.trial_text_display = tk.Label(skill_frame, text=f"trial {self.trail_cnt}", font=("Arial", 15))
        self.trial_text_display.grid(row=1, column=1, sticky="nsew")

    def create_next_button(self):
        button_frame = tk.Frame(self)
        button_frame.grid(row=2, column=1) #, pady=5, sticky=tk.EW)

        self.next_button = tk.Button(button_frame, text="Next", height=2, width=15, font=("Arial", 12), command=self.next_phase_btn_callback)
        self.next_button.pack()
        
        self.next_button.config(state='disabled')

    def create_teach_demo_button(self):
        button_frame = tk.Frame(self)
        button_frame.grid(row=2, column=0) #, pady=5, sticky=tk.EW)

        self.demonstrate_button = tk.Button(button_frame, text="Record", height=2, width=15, font=("Arial", 12), command=self.demonstrate_btn_callback)
        self.demonstrate_button.pack()
        
        self.demonstrate_button.config(state='disabled')

    def create_score_indicator(self, guidance):
        self.score_frame = tk.Frame(self)
        self.score_frame.grid(row=3, column=0)

        if guidance:
            guidance_label = tk.Label(self.score_frame,text='Score is :')
            guidance_label.grid(row=0, column=0)

            self.score_text = tk.Text(self.score_frame, width=30, height=1, wrap=tk.WORD)
            self.score_text.grid(row=0, column=1)  # Adjusted pady

            self.score_text.insert(tk.END, '\n          ')
            self.score_text.see(tk.END)
        else:
            no_guidance_label1 = tk.Label(self.score_frame,width=21, height=1, text='               ')
            no_guidance_label1.grid(row=0, column=0)

            no_guidance_label2 = tk.Label(self.score_frame,width=21, height=1, text='               ')
            no_guidance_label2.grid(row=0, column=1)

    def mode_selection(self, skill_var):
        self.reset_teaching_config()
        self.unlock_slider()
        self.demonstrate_button.config(state='normal')

        self.position_slider_x.set(0.0)
        self.position_slider_y.set(0.0)
        self.velocity_slider_x.set(0.0)
        self.velocity_slider_y.set(0.0)
        if self.force_teach_flag:
            self.force_slider_x.set(0.0)
            self.force_slider_y.set(0.0)

        if skill_var == 1:
            self.mode = 'S1NG'
            self.L_set(1)
        elif skill_var == 2:
            self.mode = 'S2NG'
            self.L_set(2)
        elif skill_var == 3:
            self.mode = 'S1G'
            self.L_set(1)
        elif skill_var == 4:
            self.mode = 'S2NG'
            self.L_set(2)
            self.score_frame.destroy()
            self.create_score_indicator(False)
        elif skill_var == 5:
            self.mode = 'S1NG'
            self.L_set(1)
        elif skill_var == 6:
            self.mode = 'S2NG'
            self.L_set(2)   

        self.simulator.robot.set_L(self.L)
        
        self.demo()
        self.simulator.draw_arm(ani=False, show_arm=False, show_start_pt=self.show_strating_point)

        if self.mode == 'S1G':
            self.create_score_indicator(True)
        elif self.mode == 'S1NG' or self.mode == 'S2NG':
            pass

    def lock_slider(self):
        self.position_slider_x.config(state="disabled")
        self.position_slider_y.config(state="disabled")
        self.velocity_slider_x.config(state="disabled")
        self.velocity_slider_y.config(state="disabled") 
        if self.force_teach_flag:
            self.force_slider_x.config(state="disabled") 
            self.force_slider_y.config(state="disabled") 

    def unlock_slider(self):
        self.position_slider_x.config(state="normal")
        self.position_slider_y.config(state="normal")
        self.velocity_slider_x.config(state="normal")
        self.velocity_slider_y.config(state="normal")
        if self.force_teach_flag:
            self.force_slider_x.config(state="normal") 
            self.force_slider_y.config(state="normal") 

    def demo(self):
        if (self.target_point[0] - 0) ** 2 + (self.target_point[1] - 0) ** 2 <= 2 ** 2:
            for i in range(self.step):
                self.simulator.robot.trajectory.append(list(self.simulator.robot.get_end_effector_position()))
                self.simulator.robot.velocity.append(list(self.simulator.robot.get_end_effector_velocity()))
                
                self.simulator.robot.update_rk4()

            self.save_desired_dynamic_motion()
            self.teacher_trajectory = self.simulator.robot.trajectory
            self.simulator.reset_robot()

    def save_desired_dynamic_motion(self):
        trajectory = np.array(self.simulator.robot.trajectory)
        velocity = np.array(self.simulator.robot.velocity)

        self.actual_q1 = self.simulator.robot.joint1_angle_track
        self.actual_q2 = self.simulator.robot.joint2_angle_track
        self.actual_q1_dot = self.simulator.robot.joint1_velocity_track
        self.actual_q2_dot = self.simulator.robot.joint2_velocity_track
        self.actual_x = trajectory[:, 0]
        self.actual_y = trajectory[:, 1]
        self.actual_x_dot = velocity[:, 0]
        self.actual_y_dot = velocity[:, 1]

    def x_pos_slider_changed(self, value):
        # Convert the slider value to a suitable range for the x position
        x_position = float(value) / 10000
        # Calculate the maximum allowed radius
        max_radius = 2
        
        # Calculate the squared distance from the origin
        distance_squared = x_position**2 + (self.position_slider_y.get() / 10000)**2
            
        if distance_squared > max_radius**2:
            # Calculate the angle between the desired position and the positive x-axis
            angle = math.atan2(self.position_slider_y.get() / 10000, x_position)
            
            # Scale back the x position to lie on the circle with the maximum radius
            y_position = max_radius * math.sin(angle)
            self.position_slider_y.set(y_position*10000)
        
        # Calculate the joint angles that will result in the desired end-effector position
        joint_angles = self.simulator.robot.inverse_kinematics(x_position, self.position_slider_y.get() / 10000)
        # Update the robot's joint angles
        self.simulator.robot.joint1_angle, self.simulator.robot.joint2_angle = joint_angles[0], joint_angles[1]
        # Redraw the robot
        self.simulator.draw_arm(self.velocity_slider_x.get() / 5000, self.velocity_slider_y.get() / 5000, self.force_slider_x.get() / 5000, self.force_slider_y.get() / 5000, ani=False, show_start_pt=self.show_strating_point)


    def y_pos_slider_changed(self, value):
        # Convert the slider value to a suitable range for the y position
        y_position = float(value) / 10000
        # Calculate the maximum allowed radius
        max_radius = 2
        
        # Calculate the squared distance from the origin
        distance_squared = (self.position_slider_x.get() / 10000)**2 + y_position**2
        
        if distance_squared > max_radius**2:
            # Calculate the angle between the desired position and the positive x-axis
            angle = math.atan2(y_position, self.position_slider_x.get() / 10000)
            
            # Scale back the y position to lie on the circle with the maximum radius
            x_position = max_radius * math.cos(angle)
            self.position_slider_x.set(x_position*10000)

        # Calculate the joint angles that will result in the desired end-effector position
        joint_angles = self.simulator.robot.inverse_kinematics(self.position_slider_x.get() / 10000, y_position)
        # Update the robot's joint angles
        self.simulator.robot.joint1_angle, self.simulator.robot.joint2_angle = joint_angles[0], joint_angles[1]
        # Redraw the robot
        self.simulator.draw_arm(self.velocity_slider_x.get() / 5000, self.velocity_slider_y.get() / 5000, self.force_slider_x.get() / 5000, self.force_slider_y.get() / 5000, ani=False, show_start_pt=self.show_strating_point)

    def x_vel_slider_changed(self, value):
        # Convert the slider value to a suitable range for the x position
        x_velocity = float(value) / 5000
        # Keep the y position the same
        y_velocity = self.velocity_slider_y.get() / 5000
        # Calculate the joint angles that will result in the desired end-effector position
        joint_velocity = self.simulator.robot.get_joint_velocity_from_ee([x_velocity, y_velocity])
        # Update the robot's joint angles
        self.simulator.robot.joint1_velocity, self.simulator.robot.joint2_velocity = joint_velocity[0], joint_velocity[1]
        # Redraw the robot
        self.simulator.draw_arm(self.velocity_slider_x.get() / 5000, self.velocity_slider_y.get() / 5000, self.force_slider_x.get() / 5000, self.force_slider_y.get() / 5000, ani=False, show_start_pt=self.show_strating_point)
        
    def y_vel_slider_changed(self, value):
        # Convert the slider value to a suitable range for the x position
        y_velocity = float(value) / 5000
        # Keep the y position the same
        x_velocity = self.velocity_slider_x.get() / 5000
        # Calculate the joint angles that will result in the desired end-effector position
        joint_velocity = self.simulator.robot.get_joint_velocity_from_ee([x_velocity, y_velocity])
        # Update the robot's joint angles
        self.simulator.robot.joint1_velocity, self.simulator.robot.joint2_velocity = joint_velocity[0], joint_velocity[1]
        # Redraw the robot
        self.simulator.draw_arm(self.velocity_slider_x.get() / 5000, self.velocity_slider_y.get() / 5000, self.force_slider_x.get() / 5000, self.force_slider_y.get() / 5000, ani=False, show_start_pt=self.show_strating_point)
        
    def x_force_slider_changed(self, value):
        self.force_slider_value_x.config(text=f"{float(value)/5000}")
        self.simulator.draw_arm(self.velocity_slider_x.get() / 5000, self.velocity_slider_y.get() / 5000, self.force_slider_x.get() / 5000, self.force_slider_y.get() / 5000, ani=False, show_start_pt=self.show_strating_point)
        
    def y_force_slider_changed(self, value):
        self.force_slider_value_y.config(text=f"{float(value)/5000}")
        self.simulator.draw_arm(self.velocity_slider_x.get() / 5000, self.velocity_slider_y.get() / 5000, self.force_slider_x.get() / 5000, self.force_slider_y.get() / 5000, ani=False, show_start_pt=self.show_strating_point)
        
    def next_phase_btn_callback(self):
        if self.trail_cnt < self.max_trail_num: # trail 1, trial 2 
            self.trail_cnt += 1
        else:                                   # trial 3
            if self.phase_cnt <= 6:
                self.phase_cnt += 1
            self.trail_cnt = 1                  # reset trail = 1
        self.trial_text_display.config(text=f"trial {self.trail_cnt}")

        self.next_button.config(state='disabled')

        if self.phase_cnt == 2:
            self.text_display.config(text="Phase 2: Please teach Skill 2", font=("Arial", 20))
        elif self.phase_cnt == 3:
            self.text_display.config(text="Phase 3: Please teach Skill 1", font=("Arial", 20))
        elif self.phase_cnt == 4:
            self.text_display.config(text="Phase 4: Please teach Skill 2", font=("Arial", 20))
        elif self.phase_cnt == 5:
            self.text_display.config(text="Phase 5: Please teach Skill 1", font=("Arial", 20))
        elif self.phase_cnt == 6:
            self.text_display.config(text="Phase 6: Please teach Skill 2", font=("Arial", 20))
        print("Phase {} - {}".format(self.phase_cnt, self.trail_cnt))
        
        self.reset_canvas_config()
        self.mode_selection(skill_var=self.phase_cnt)

        self.unlock_slider()

    def demonstrate_btn_callback(self):
        # self.unlock_slider()
        if self.demo_cnt <= self.demo_num-1:
            # self.phi_each = [self.position_slider_x.get()/10000, self.position_slider_y.get()/10000, self.velocity_slider_x.get()/5000, self.velocity_slider_y.get()/5000]
            # self.phi_recording.append(self.phi_each)
            # self.phi_each = [self.position_slider_x.get()/10000, self.position_slider_y.get()/10000, self.velocity_slider_x.get()/5000, self.velocity_slider_y.get()/5000, 1]

            self.phi_recording.append([self.position_slider_x.get()/10000, self.position_slider_y.get()/10000, self.velocity_slider_x.get()/5000, self.velocity_slider_y.get()/5000])
            if self.force_teach_flag:
                self.force_recording.append([self.force_slider_x.get()/5000, self.force_slider_y.get()/5000])
            self.demo_cnt += 1
            if self.demo_cnt == self.demo_num:
                self.lock_slider()
                messagebox.showinfo("Information", "This is what we learned.")
                self.demonstrate_button.config(state='disabled')
                self.next_button.config(state='normal')
                # self.teach_button.config(state='normal')
                self.teaching()
                self.lock_slider()
                return
        
    def teaching(self):
        if len(self.phi_recording) == 0:
            messagebox.showwarning('Warning', 'No demonstration has recorded.')
            return
        elif len(self.phi_recording) !=0 and len(self.phi_recording) != 5:
            messagebox.showwarning('Warning', 'Incomplete demonstration.')
            return

        phi = np.vstack((np.array(self.phi_recording).T, np.ones((1, 5))))

        if self.force_teach_flag == False: # force from calculated with noise
            force_real = self.L @ phi
            force_noi_diff = self.force_std_noise * np.random.randn(2, 5)
            force_with_noise = force_real + force_noi_diff
            force_given = force_with_noise
        elif self.force_teach_flag == True:
            force_teached = np.array(self.force_recording).T # np.shape (5,2)
            force_given = force_teached

        # calculate theta
        self.L_learner = self.simulator.learn(phi, force_given)
        print(f"{self.mode}: self.L = {self.L[0]}, {self.L[1]}")
        print(f"{self.mode}: self.L_learner = {self.L_learner[0]}, {self.L_learner[1]}")

        self.simulator.robot.set_L(self.L_learner)
        # show student's animation
        self.next_button.config(state='disabled')
        if '1' in self.mode:
            self.simulator.robot.joint1_angle, self.simulator.robot.joint2_angle = self.simulator.robot.inverse_kinematics(0.8, 1.1)
        elif '2' in self.mode:
            self.simulator.robot.joint1_angle, self.simulator.robot.joint2_angle = self.simulator.robot.inverse_kinematics(0.2, 0.3)
        self.simulator.robot.joint1_velocity, self.simulator.robot.joint2_velocity = 0.0, 0.0
        for i in range(self.step):
            self.simulator.robot.trajectory.append(list(self.simulator.robot.get_end_effector_position()))
            self.simulator.robot.velocity.append(list(self.simulator.robot.get_end_effector_velocity()))
            if True: # true = show student's animation, false = dont show
                self.simulator.draw_arm()
            self.simulator.robot.update_rk4()
        self.student_trajectory = self.simulator.robot.trajectory

        # calculate error
        rmse = np.sqrt(np.mean((np.array(self.teacher_trajectory) - np.array(self.student_trajectory))**2))
        nmse = self.nmse_cal(self.L, self.L_learner)
        el2 = self.el2_cal(self.L, self.L_learner)
        print(f"rmse: {rmse}, nmse: {nmse}, el2: {el2}")

        # calculate score for phase 3, and display only for phase 3
        score1 = self.score1(phi, force_given)
        new_score = self.new_score(phi, force_given)
        score5 = self.score5(phi, force_given)
        print(f"score1: {score1}, new_score: {new_score}, score5: {score5}")

        if self.mode == 'S1G':
            guidance_score = score5
            self.score_text.insert(tk.END, '\n {}.'.format(guidance_score)) # display score
            self.score_text.see(tk.END)

        # store data tocsv file
        self.store_data_tocsv(self.phase_cnt, self.trail_cnt, phi, force_given, self.L_learner, rmse, nmse, el2, new_score)

        # reset robot
        self.simulator.reset_robot()
        self.reset_teaching_config()
        self.next_button.config(state='normal')

        # when phase_cnt = 6, exp. finished.
        if self.phase_cnt >= 6 and self.trail_cnt == 3:
            print("Finished!")
            messagebox.showinfo("Information", "Experiment Finished.")
            self.quit()

    def nmse_cal(self, theta_real, theta_learned):
        test_phi = np.array([[1.54547433, 1.82535471, 0.20575465, 0.40244344, 0.09569714, 1.04273663, 1.72299245, 0.95102198, 0.68314063, 0.61546897],
            [0.08959529, 0.01713778, 1.64400987, 0.53837856, 0.17259851, 1.47867964, 0.31134094, 0.59224916, 0.6820265, 1.03555592],
            [-0.12833334, -0.10372383, -0.00663499, 0.1608878, -0.10024024, -0.26926053, -0.40315198, 0.23943124, -0.1348775, 0.05278681],
            [0.15063608, 0.39135257, 0.16674541, -0.07797693, 0.38231418, 0.25929959, -0.25023862, 0.44598129, -0.46304699, -0.47250088],
            [1,1,1,1,1,1,1,1,1,1]])
        n = test_phi.shape[1]
        
        force_desired = np.dot(theta_real, test_phi)
        force_learned = np.dot(theta_learned, test_phi)

        force_mse = np.mean((force_learned-force_desired)**2)
        nmse = force_mse / np.var(force_desired, ddof=0)
        return nmse

    def el2_cal(self, theta_real, theta_learned):
        el2 = np.linalg.norm(theta_real - theta_learned)
        return el2

    def store_data_tocsv(self, phase, trail, phi, force, theta_learned, rmse, nmse, el2, score):
        # set path
        current_path = os.path.dirname(os.path.abspath(__file__))
        # new_folder_path = os.path.join(current_path, f'pilot_exp_results/pilot_exp_id_{self.participant_number}')
        new_folder_path = os.path.join(current_path, f'pilot_exp_results')
        os.makedirs(new_folder_path, exist_ok=True) # creates the subfolder if it does not exist
        csv_file_name  = f"pilot_exp_id_{self.participant_number}.csv"
        csv_file_path = os.path.join(new_folder_path, csv_file_name)
        
        # store value
        with open(csv_file_path, mode="a", newline="") as csvfile:
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
            csvwriter.writerows([["nmse", f"{nmse}"]])
            csvwriter.writerows([["el2", f"{el2}"]])
            csvwriter.writerow([])

            if self.mode == 'S1G':
                csvwriter.writerows([["score", f"{score}"]])
                csvwriter.writerow([])

        print("Matrix saved to", csv_file_name)
        print("-----------------------------------------")  
    
    def SVD(self, phi):
        _, S, _ = np.linalg.svd(phi)
        return S
    
    def score(self, phi):
        sigma_min = np.min(self.SVD(phi))
        sigma_min_inverse = 1 / (sigma_min + 1e-10)

        if sigma_min_inverse <= 50:
            score = 99
        elif sigma_min_inverse > 50 and sigma_min_inverse <=200:
            score = 90
        elif sigma_min_inverse > 200 and sigma_min_inverse <= 400:
            score = 80
        elif sigma_min_inverse > 400 and sigma_min_inverse <= 600:
            score = 60
        elif sigma_min_inverse > 600 and sigma_min_inverse <= 800:
            score = 40
        elif sigma_min_inverse > 800 and sigma_min_inverse <= 1000:
            score = 20
        elif sigma_min_inverse > 1000:
            score = 0
        
        return score

    def score1(self, phi, force):    
        phi_star = np.array([[ 0.00353898,  0.04301875,  0.08119696,  0.10114971,  1.19618413],
                    [ 0.06719514,  1.69925865,  0.15236148,  1.7271737,   1.19777084],
                    [ 0.36519301, -0.88843374, -0.49570924,  0.92063472,  0.43162667],
                    [-0.4055831,  -0.40190829,  0.97655309,  0.97670281, -0.77406324],
                    [ 1.,          1.,          1.,          1.,          1.]])

        force_star = np.array([[ 1.033032,   9.68997993,  6.82637378, -3.14789218, -3.00993908],
                            [ 6.23749627,  1.31558208, -3.69295611, -8.41844078,  5.42513014]])
        
        star_value = np.vstack((phi_star, force_star))
        
        sorted_indices = np.argsort(phi[0, :]) # Sort the entire matrix column-wise using these indices
        sorted_phi = phi[:, sorted_indices]
        sorted_force = force[:, sorted_indices]
        input_value = np.vstack((sorted_phi, sorted_force))

        score = np.sum(np.abs(star_value - input_value))

        return score
    
    def new_score(self, phi, force):
        phi_star = np.array([[0.00353898, 1.19618413, 0.08119696, 0.10114971, 0.04301875],                        
                         [0.06719514, 1.19777084, 0.15236148, 1.72717370, 1.69925865],                        
                         [0.36519301, 0.43162667, -0.49570924, 0.92063472, -0.88843374],                        
                         [-0.40558310, -0.77406324, 0.97655309, 0.97670281, -0.40190829],                        
                         [1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000]])         
        force_star = np.array([[1.03303200, -3.00993908, 6.82637378, -3.14789218, 9.68997993],                        
                            [6.23749627, 5.42513014, -3.69295611, -8.41844078, 1.31558208]])        
        star_value = np.vstack((phi_star, force_star))
        sorted_matrix = self.sorted_func(phi, force)    
        score = np.sum(np.abs(star_value - sorted_matrix))    
        return score
    
    def sorted_func(self, matrix1, matrix2):
        matrix1 = np.array(matrix1)
        matrix2 = np.array(matrix2)
        matrix = np.vstack((matrix1, matrix2))
        # Calculate the L2 norm for each column
        column_l2_norms = np.linalg.norm(matrix, axis=0, ord=2)
        # Sort the columns based on L2 norms
        sorted_indices = np.argsort(column_l2_norms)
        sorted_matrix = matrix[:, sorted_indices]
        return sorted_matrix

    def fit_score_s1(self, min_sigma):
        x_value = 1/(min_sigma + 1e-12)
        # Hardcoded parameters from the fitted model
        intercept = -0.062157
        coef = 0.0001005
        x_intercept = 618.21
        x_el2_05 = 5591.02
        
        # If x is less than the intersection with the X-axis, the score is 100
        if x_value < x_intercept:
            return 100
        
        # If x is greater than the x value when el2 = 0.5, the score is 0
        elif x_value > x_el2_05:
            return 0
        
        # Otherwise, calculate the el2 value using the hardcoded model parameters
        else:
            el2_value = intercept + coef * x_value
            
            # Compute the score based on the el2 value
            fit_score = 100 * (1 - (el2_value - 0) / (0.5 - 0))
        
        return fit_score

    def fit_score_s2(self, min_sigma):
        x_val = 1/min_sigma
        coef = 0.0004
        intercept = -0.0825
        x_intercept = 222.67

        # If x is smaller than the x-intercept
        if x_val < x_intercept:
            return 100

        # Compute the predicted el2 using the regression equation
        predicted_el2 = coef * x_val + intercept

        # If predicted el2 is greater than 2
        if predicted_el2 > 2:
            return 0

        # Rescale the predicted el2 to a score between 0 and 100
        fit_score = 100 * (2 - predicted_el2) / 2

        return fit_score
    
    def plot_dynamic_motion(self):
        # Create a figure with 4 subplots
        trajectory = np.array(self.simulator.robot.trajectory)
        velocity = np.array(self.simulator.robot.velocity)

        fig, axs = plt.subplots(4, 2)
        # fig.suptitle("{}th Teaching Dynamics Motion \n\n from StartingPoint:{} to TargetPoint:{} \n\n L = {}\n         {}".format(self.teaching_nums, self.starting_point, self.target_point, self.robot.L[0], self.robot.L[1]))
        fig.suptitle("Teaching Dynamics Motion \n\n from StartingPoint:{} to TargetPoint:{} \n\n L = {}\n         {}".format(self.starting_point, self.target_point, self.robot.L[0], self.robot.L[1]))
        #fig.suptitle('Teacher\'s Dynamics Motion \n L = {}\n       {}'.format(self.robot.L[0], self.robot.L[1]))

        idx = [i for i in range(len(self.simulator.robot.joint1_angle_track))]

        axs[0, 0].plot(idx, self.actual_q1, 'red', label = 'desired behaviour', linestyle = ':')
        axs[0, 1].plot(idx, self.actual_q2, 'red',  linestyle = ':')
        axs[1, 0].plot(idx, self.actual_q1_dot, 'red',  linestyle = ':')
        axs[1, 1].plot(idx, self.actual_q2_dot, 'red',  linestyle = ':')
        axs[2, 0].plot(idx, self.actual_x, 'red',  linestyle = ':')
        axs[2, 1].plot(idx, self.actual_y, 'red',  linestyle = ':')
        axs[3, 0].plot(idx, self.actual_x_dot, 'red',  linestyle = ':')
        axs[3, 1].plot(idx, self.actual_y_dot, 'red',  linestyle = ':')
        
        
        axs[0, 0].plot(idx, self.simulator.robot.joint1_angle_track, label = 'learner\'s behaviour', linewidth = 0.8)
        axs[0, 1].plot(idx, self.simulator.robot.joint2_angle_track, linewidth = 0.8)
        axs[1, 0].plot(idx, self.simulator.robot.joint1_velocity_track, linewidth = 0.8)
        axs[1, 1].plot(idx, self.simulator.robot.joint2_velocity_track, linewidth = 0.8)
        axs[2, 0].plot(idx, trajectory[:, 0], linewidth = 0.8)
        axs[2, 1].plot(idx, trajectory[:, 1], linewidth = 0.8)
        axs[3, 0].plot(idx, velocity[:, 0], linewidth = 0.8)
        axs[3, 1].plot(idx, velocity[:, 1], linewidth = 0.8)

        axs[0, 0].set_xlabel('times', loc = 'right')
        axs[0, 0].set_ylabel('q1', loc = 'top')
        axs[0, 1].set_xlabel('times')
        axs[0, 1].set_ylabel('q2')
        axs[1, 0].set_xlabel('times')
        axs[1, 0].set_ylabel('q1_dot')
        axs[1, 1].set_xlabel('times')
        axs[1, 1].set_ylabel('q2_dot')
        axs[2, 0].set_xlabel('times')
        axs[2, 0].set_ylabel('x')
        axs[2, 1].set_xlabel('times')
        axs[2, 1].set_ylabel('y')
        axs[3, 0].set_xlabel('times')
        axs[3, 0].set_ylabel('x_dot')
        axs[3, 1].set_xlabel('times')
        axs[3, 1].set_ylabel('y_dot')

        fig.legend(loc = 'upper left')
        # Adjust the spacing between the subplots
        fig.tight_layout()

        # Show the plot
        plt.show(block=False)
        plt.pause(0.001) 

    def score5(self, phi, force):
        if self.phase_cnt == 1 or self.phase_cnt == 3 or self.phase_cnt == 5:
            # phi_star = np.array([[0.80, 1.02, 1.57, 1.36, 1.23],
            #                     [1.10, 1.24, 1.20, 0.72, 0.58],
            #                     [0.0, 0.90, 0.15, -0.56, -0.02],
            #                     [0.0, 0.45, -0.97, -0.66, -0.03],
            #                     [1,1,1,1,1]])
            # force_star = np.array([[1.6,  2.04, 3.14, 2.72, 2.46],
            #                        [2.2,  2.48, 2.4,  1.44, 1.16]])
            phi_star = np.array([[ 1.23,  0.8,   1.36,  1.02,  1.57],
                                [ 0.58,  1.1,   0.72,  1.24,  1.2 ],
                                [-0.02,  0.,   -0.56,  0.9,   0.15],
                                [-0.03,  0.,   -0.66,  0.45, -0.97],
                                [ 1.,    1.,    1.,    1.,    1.  ]])
            
            force_star = np.array([[ 2.46,  1.6,   2.72,  2.04,  3.14],
                                [ 1.16,  2.2,   1.44,  2.48,  2.4 ]])
        elif self.phase_cnt == 2 or self.phase_cnt == 4 or self.phase_cnt == 6:
            # phi_star = np.array([[0.21, 0.47, 0.71, 0.90, 0.15],
            #                     [0.31, 0.51, 0.69, 0.89, 0.16],
            #                     [0.21, 0.37, 0.23, 0.13, 0.02],
            #                     [0.20, 0.25, 0.22, 0.15, 0.02],
            #                     [1,1,1,1,1]])
            # force_star = np.array([[ 1.5,  -0.4,  -0.14, -0.01,  3.01],
            #                        [ 1.27,  0.32, -0.01, -0.12,  2.98]])
            phi_star = np.array([[0.47,  0.71,  0.9,   0.21,  0.15],
                                [ 0.51,  0.69,  0.89,  0.31,  0.16],
                                [ 0.37,  0.23,  0.13,  0.21,  0.02],
                                [ 0.25,  0.22,  0.15,  0.2,   0.02],
                                [ 1.,    1.,    1.,    1.,    1.  ]])
            force_star = np.array( [[-0.4,  -0.14, -0.01,  1.5,   3.01],
                                  [ 0.32, -0.01, -0.12,  1.27,  2.98]])

        star_value = np.vstack((phi_star, force_star))
        sorted_matrix = self.sorted_func(phi, force)    
        score = np.sum(np.abs(star_value - sorted_matrix))    
        return score
    

    

if __name__ == "__main__":
    robot = BasicRobotArm(link1_length=1, link2_length=1, 
                        link1_mass=1, link2_mass=1, 
                        joint1_angle=0.0, joint2_angle=0.0, 
                        joint1_velocity=0.0, joint2_velocity=0.0, 
                        joint1_torque=0.0, joint2_torque=0.0,
                        time_step=0.1, g=9.81)

    x0, y0 = 0.2, 1.5
    xt, yt = 1.5, 0.3

    app = RobotArmApp(robot=robot, demo_num=5, trial_num=3, force_teach=True, std=0.0)
    app.mainloop()