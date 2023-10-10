import os
import sys
import csv
import json
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

from SingleArm import SingleRobotArm

joint_image = mpimg.imread("./RobotConfigs/Joint.png")
link_image = mpimg.imread("./RobotConfigs/Link.png")
# Inside the class initialization
end_effector_image = mpimg.imread("./RobotConfigs/EndEffector.png")
base_image = mpimg.imread("./RobotConfigs/Base.png")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class RobotArmSimulator:
    def __init__(self, master, robot: SingleRobotArm):
        self.master = master
        self.robot = robot
        self.arm_animation = False

        self.init_q0 = 0.0

        #setting 1
        self.max_x_vel, self.max_y_vel = 0.3, 0.3 # m/s
        self.max_x_force, self.max_y_force = 1.0, 1.0 # N

        self.positions = []
        self.velocities = []
        self.accelerations = []
        self.joint_angles = []
        self.joint_velocities = []
        self.joint_accelerations = []

        self.fig = Figure(figsize=(6, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)

        self.fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

        self.ax.set_xlim(self.robot.x_min - 0.1, self.robot.x_max + 0.5)
        self.ax.set_ylim(self.robot.y_min - 0.1, self.robot.y_max + 0.5)
        self.ax.set_aspect('equal', 'box')
        self.ax.grid(True)

        self.starting_point, = self.ax.plot([], [], 'g*', ms=5)
        self.target_point, = self.ax.plot([], [], 'r*', ms=5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # initialize starting point
        self.robot.joint1_angle = self.init_q0
        x, y = self.robot.forward_kinematics_from_states_2_joints(self.robot.joint1_angle)

        if self.arm_animation == False:
            self.arm_line, = self.ax.plot([], [], 'o-', lw=3)
            # self.arm_line.set_data([0, x[0]], [0, y[0]])
            self.arm_line.set_data([0, x], [0, y])

        self.arrow = self.ax.annotate('', xy=(x + 2, y + 1), xytext=(x,y),arrowprops=dict(arrowstyle= '-|>',
                                                                                             color='blue', lw=0.5, ls='solid'))
        self.force_arrow = self.ax.annotate('', xy=(x + 2, y + 1), xytext=(x,y),arrowprops=dict(arrowstyle= '-|>',
                                                                                             color='red', lw=0.5, ls='solid'))
        self.draw_arm(ani=False)
        
    def reset_robot(self):
        # self.robot.joint1_angle = self.robot.inverse_kinematics(self.init_x0, self.init_y0)
        self.robot.joint1_angle = self.init_q0
        self.robot.joint1_velocity = 0.0
        
        self.robot.trajectory = []
        self.robot.velocity = []
        self.robot.joint1_angle_track = []
        self.robot.joint1_velocity_track = []
        
    def draw_arm(self, vel_len_q=0, force_len_q=0, ani=True, show_arm=True):
        x, y = self.robot.forward_kinematics_from_states_2_joints(self.robot.joint1_angle)

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
                self.joint1_image_display.set_extent([x-0.1, x+0.1, y-0.1, y+0.1])
            else:
                self.joint1_image_display = self.ax.imshow(joint_image, extent=[x-0.1, x+0.1, y-0.1, y+0.1])

            if hasattr(self, "joint2_image_display"):
                self.joint2_image_display.set_data(joint_image)
                self.joint2_image_display.set_extent([x[1]-0.1, x[1]+0.1, y[1]-0.1, y[1]+0.1])
            else:
                self.joint2_image_display = self.ax.imshow(joint_image, extent=[x[1]-0.1, x[1]+0.1, y[1]-0.1, y[1]+0.1])

            # Calculate the length and orientation for the first link
            length1 = np.sqrt(x**2 + y**2)
            angle1 = np.arctan2(y, x) * (180/np.pi)

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
            self.arm_line.set_data([0, x], [0, y])
            if not show_arm:
                self.arm_line.set_data([], [])

        # self.draw_arrow(vel_len_x, vel_len_y, force_len_x, force_len_y)
        self.draw_arrow(vel_len_q, force_len_q)
        self.fig.canvas.draw()
        if ani:
            self.fig.canvas.flush_events()

    def draw_arrow(self, vel_len_q, force_len_q):
        x, y = self.robot.forward_kinematics_from_states_2_joints(self.robot.joint1_angle)
        q1 = self.robot.joint1_angle

        self.arrow.remove()
        self.force_arrow.remove()
        
        start_x = x
        start_y = y
        end_x = x - vel_len_q * np.sin(q1)
        end_y = y + vel_len_q * np.cos(q1)
        
        force_end_x = x + force_len_q * np.sin(q1) * 0.5
        force_end_y = y - force_len_q * np.cos(q1) * 0.5

        self.arrow = self.ax.annotate('', xy=(end_x, end_y), xytext=(start_x,start_y), arrowprops=dict(arrowstyle= '-|>',
                                                                                             color='blue', lw=0.8, ls='solid'))
        
        self.force_arrow = self.ax.annotate('', xy=(force_end_x, force_end_y), xytext=(start_x,start_y), arrowprops=dict(arrowstyle= '-|>',
                                                                                             color='red', lw=1.0, ls='solid'))

    def learn(self, phi, force, demo=5):
        lam = 1e-6
        I = np.identity(demo)

        learned = (np.linalg.inv(phi @ phi.T + lam * I) @ phi @ (force.T)).T

        return learned
    
class RobotArmApp(tk.Tk):
    def __init__(self, robot: SingleRobotArm, demo_num, trial_num, force_teach, std, show_guidace_flag, student_animation):
        super().__init__()
        self.title("Two-Axis RR Robot Arm Simulator")
        self.geometry("1150x650") # width x height

        self.robot = robot

        # self.target_point = [1.237, 0.592]
        self.target_point = [1.22857732, 0.58413413]
        self.L_real = np.zeros((2, 5))
        self.L_learner = np.zeros((2, 5))
        self.step = 0        

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
        # self.max_trail_num = 1
        self.phase_cnt = 1              # 6 phase in total
        self.max_phase_num = 6

        self.force_teach_flag = force_teach
        self.std_noise = std
        self.guidancetype = show_guidace_flag

        self.student_animation_show_flag = student_animation

        # data save initialization
        self.records = {
            'datas': [],
            'count': 0
        }
        
    def reset_canvas_config(self):        

        self.simulator.reset_robot()

        self.simulator.draw_arm(ani=False)

    def reset_teaching_config(self):

        self.demo_cnt = 0

        self.phi_recording = []
        self.force_recording = []
        
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
        self.expstart_frame.grid(row=2, column=0)

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
                self.create_formal_teaching_callback_page()
                return

        minutes, seconds = divmod(remaining_time.seconds, 60)
        countdown_text = f"{minutes:02}:{seconds:02}"

        if self.clock_show_flag == 1:
            self.countdown_label.config(text=countdown_text)

        self.after(1000, self.update_countdown_clock, target_time)

    def create_canvas(self):
        self.plot_frame = tk.Frame(self)
        self.plot_frame.grid(row=0, column=2, rowspan=4, padx=20, pady=5)
        self.simulator = RobotArmSimulator(self.plot_frame, self.robot)
        self.simulator.draw_arm(ani=False)

    def create_play_mode(self):
        self.play_mode_frame = tk.Frame(self)
        self.play_mode_frame.grid(row=0, column=0, padx = 104, pady = 20)

        self.skill1_demo_btn = tk.Button(self.play_mode_frame, text="Skill 1 Movement", height=2, width=33, font=("Arial", 12), command=self.skill1_demo_callback)
        self.skill1_demo_btn.grid(row=0, column=0, sticky="nsew", pady = 20)

        self.skill2_demo_btn = tk.Button(self.play_mode_frame, text="Skill 2 Movement", height=2, width=33, font=("Arial", 12), command=self.skill2_demo_callback)
        self.skill2_demo_btn.grid(row=1, column=0, sticky="nsew") #pady = 5)

    def L_set(self, skill_num):
        if skill_num == 1:
            # self.target_point = [1.237, 0.592] 
            self.target_point = [0, 1]
            # self.L_real = np.array([[1, 0, 0, 0, 0],
            #             [0, 1, 0, 0, 0]])
            # self.L_real = np.array([[0.3, 0, 0, 0, 0],
            #             [0, 0.3, 0, 0, 0]])
            self.L_real = np.array([[1], [0]])
            # self.step = 75
            self.step = 160

            self.simulator.reset_robot()
        elif skill_num == 2:
            self.target_point = [1, 0]
            # self.L_real = np.array([[-1, 0, -2, 0, 0.8],
            #                    [0, -1, 0, -2, 1.2]])
            self.L_real = np.array([[-1], [+1]])
            # self.step = 153
            # self.step = 138
            self.step = 200

            self.simulator.reset_robot()
        else:
            raise ValueError('Invalid Skill Number.')

    def skill1_demo_callback(self):
        self.skill1_demo_btn.config(state='disabled')
        self.skill2_demo_btn.config(state='disabled')
        # self.simulator.init_x0, self.simulator.init_y0 = 0.8, 1.1
        # self.simulator.init_x0, self.simulator.init_y0 = 0, 1
        self.simulator.init_q0 = 0
        self.simulator.reset_robot()
        self.L_set(1)
        self.simulator.robot.set_L(self.L_real)

        self.simulator.starting_point.set_data(0, 1)
        self.simulator.target_point.set_data(1, 0)

        for i in range(self.step):
            self.simulator.robot.trajectory.append(list(self.simulator.robot.get_end_effector_position()))
            self.simulator.robot.velocity.append(list(self.simulator.robot.get_end_effector_velocity()))

            self.simulator.draw_arm()
            
            self.simulator.robot.update()

        self.simulator.starting_point.set_data([], [])
        self.simulator.target_point.set_data([], [])
        self.skill1_demo_btn.config(state='normal')
        self.skill2_demo_btn.config(state='normal')

        self.simulator.init_x0, self.simulator.init_y0 = 0, 1
        self.simulator.reset_robot()

    def skill2_demo_callback(self):
        self.skill1_demo_btn.config(state='disabled')
        self.skill2_demo_btn.config(state='disabled')
        self.L_set(2)

        self.simulator.target_point.set_data([0.8], [1.2])

        starting_joint_pos = 0
        
        self.simulator.init_x0, self.simulator.init_y0 = starting_joint_pos
        self.simulator.starting_point.set_data([self.simulator.init_x0], [self.simulator.init_y0])
        self.simulator.reset_robot()
        self.simulator.robot.set_L(self.L_real)

        x, y = self.simulator.robot.get_end_effector_position()
        while (x-self.target_point[0])**2 + (y-self.target_point[1])**2 > 0.0005:
            self.simulator.robot.trajectory.append(list(self.simulator.robot.get_end_effector_position()))
            self.simulator.robot.velocity.append(list(self.simulator.robot.get_end_effector_velocity()))

            self.simulator.draw_arm()
            
            self.simulator.robot.update()

            x, y = self.simulator.robot.get_end_effector_position()
        
        # self.simulator.init_x0, self.simulator.init_y0 = 0.2, 0.5
        self.simulator.init_x0, self.simulator.init_y0 = 0.2111, 0.3111
        self.simulator.starting_point.set_data([], [])
        self.simulator.target_point.set_data([], [])
        self.simulator.reset_robot()

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
        
    def create_formal_teaching_callback_page(self):
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
            self.demonstrate_button.config(state='normal')
        else:
            messagebox.showerror('Please input a valid participant ID.')
    
    def create_slider(self):
        self.slide_frame = tk.Frame(self)
        self.slide_frame.grid(row=1, column=0, columnspan = 2)

        # x position slider label
        pos_slider_label_q = tk.Label(self.slide_frame,text='Position, q:   min', fg="green")
        pos_slider_label_q.grid(row=0, column=0)
        self.position_slider_q = tk.Scale(self.slide_frame, from_= 0*10000, to=self.simulator.robot.x_max*10000, orient='horizontal', 
                                        command=self.q_pos_slider_changed, state='disabled', 
                                        length= 300, showvalue=0)
        self.position_slider_q.grid(row=0, column=1, pady =15)
        
        # x slider value
        self.pos_val_q = tk.StringVar()
        self.pos_val_q.set('max')
        self.pos_label_q = tk.Label(self.slide_frame, textvariable=self.pos_val_q, width=10, fg="green")
        self.pos_label_q.grid(row=0, column=2)

        

        # xdot velocity slider label
        vel_slider_label_q = tk.Label(self.slide_frame,text='Velocity, x:   min', fg="blue")
        vel_slider_label_q.grid(row=2, column=0)
        self.velocity_slider_q = tk.Scale(self.slide_frame, from_=-self.simulator.max_x_vel*5000, to=self.simulator.max_x_vel*5000, orient='horizontal', 
                                          command=self.q_vel_slider_changed, state='disabled', 
                                          length= 300, showvalue=0)
        self.velocity_slider_q.grid(row=2, column=1, pady=15)
        # xdot slider value
        self.vel_val_q = tk.StringVar()
        self.vel_val_q.set("max")
        self.vel_label_q = tk.Label(self.slide_frame, textvariable=self.vel_val_q,  fg="blue")
        self.vel_label_q.grid(row=2, column=2)

        


        if self.force_teach_flag:
            # x force slider label
            force_slider_label_q = tk.Label(self.slide_frame,text='Force, x:   min', fg="red")
            force_slider_label_q.grid(row=4, column=0, padx=20)
            self.force_slider_q = tk.Scale(self.slide_frame, from_=-self.simulator.max_x_force*5000, to=self.simulator.max_x_force*5000, orient='horizontal', 
                                            command=self.torque_slider_changed, state='disabled', 
                                            length= 300, showvalue=0)
            self.force_slider_q.grid(row=4, column=1, pady=15)
            # x force slider value
            self.force_val_q = tk.StringVar()
            self.force_val_q.set("max")
            self.force_val_q = tk.Label(self.slide_frame, textvariable=self.force_val_q, fg="red")
            self.force_val_q.grid(row=4, column=2)
            self.force_slider_value_q = tk.Label(self.slide_frame,text='None', fg="red")
            self.force_slider_value_q.grid(row=4, column=3)
            self.force_slider_value_q.grid_forget()


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
        self.next_button.config(state='disabled') # 初始化next button就是disabled的

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

        self.position_slider_q.set(self.simulator.init_q0*10000)
        self.velocity_slider_q.set(0.0)
        if self.force_teach_flag:
            self.force_slider_q.set(0.0)

        if skill_var in [1,3,5]:
            self.simulator.starting_point.set_data(0.8, 1.1)
            # self.simulator.target_point.set_data(1.237, 0.592)
            self.simulator.target_point.set_data(1.22857732, 0.58413413)
            self.L_set(1)
        elif skill_var in [2,4,6]:
            # self.simulator.starting_point.set_data(0.2, 0.3)
            self.simulator.starting_point.set_data([], []) # dont show starting point, 因为这是从任意一个点。
            self.simulator.target_point.set_data(0.8, 1.2)
            self.L_set(2)
            if skill_var == 4:
                if self.guidancetype:
                    self.score_frame.destroy()
                    self.create_score_indicator(False)

        self.simulator.robot.set_L(self.L_real)

        self.teacher_demo()
        self.simulator.draw_arm(ani=False, show_arm=False)

        if self.phase_cnt == 3 and self.guidancetype:
            self.create_score_indicator(True)
        else:
            pass
        
    def next_phase_btn_callback(self):

        self.unlock_slider()

        self.position_slider_q.set(self.simulator.init_x0*10000)
        self.velocity_slider_q.set(0.0)
        if self.force_teach_flag:
            self.force_slider_q.set(0.0)

        self.next_button.config(state='disabled')
        self.demonstrate_button.config(state='normal')

        if self.trail_cnt < self.max_trail_num: # trail 1, trial 2 
            self.trail_cnt += 1
        else:                                   # trial 3
            if self.phase_cnt <= 6:
                self.phase_cnt += 1
            self.trail_cnt = 1                  # reset trail = 1
        self.trial_text_display.config(text=f"trial {self.trail_cnt}")

        # phase change -> change skill 
        if self.phase_cnt in [1, 3, 5] and self.trail_cnt == 1:
            self.mode_selection(skill_var=self.phase_cnt)
            self.text_display.config(text=f"Phase {self.phase_cnt}: Please teach skill 1", font=("Arial", 20))
            print("reset all skill's parameter, phase and trail :", self.phase_cnt, self.trail_cnt)
        elif self.phase_cnt in [2, 4, 6] and self.trail_cnt == 1:
            self.mode_selection(skill_var=self.phase_cnt)
            self.text_display.config(text=f"Phase {self.phase_cnt}: Please teach skill 2", font=("Arial", 20))
            print("reset all skill's parameter, phase and trail :", self.phase_cnt, self.trail_cnt)
        print("Phase {} - {}".format(self.phase_cnt, self.trail_cnt))
                
        self.reset_canvas_config()
    
    def lock_slider(self):
        self.position_slider_q.config(state="disabled")
        self.velocity_slider_q.config(state="disabled")
        if self.force_teach_flag:
            self.force_slider_q.config(state="disabled") 

    def unlock_slider(self):
        self.position_slider_q.config(state="normal")
        self.velocity_slider_q.config(state="normal")
        if self.force_teach_flag:
            self.force_slider_q.config(state="normal") 

    def teacher_demo(self):
        if (self.target_point[0] - 0) ** 2 + (self.target_point[1] - 0) ** 2 <= 2 ** 2:
            for i in range(self.step):
                self.simulator.robot.trajectory.append(list(self.simulator.robot.get_end_effector_position()))
                self.simulator.robot.velocity.append(list(self.simulator.robot.get_end_effector_velocity()))
                
                self.simulator.robot.update()

            self.save_desired_dynamic_motion()
            self.teacher_trajectory = self.simulator.robot.trajectory
            self.simulator.reset_robot()
            print(np.shape(self.teacher_trajectory))

    def save_desired_dynamic_motion(self):
        trajectory = np.array(self.simulator.robot.trajectory)
        velocity = np.array(self.simulator.robot.velocity)

        self.actual_q1 = self.simulator.robot.joint1_angle_track
        self.actual_q1_dot = self.simulator.robot.joint1_velocity_track
        self.actual_x = trajectory[:, 0]
        self.actual_y = trajectory[:, 1]
        self.actual_x_dot = velocity[:, 0]
        self.actual_y_dot = velocity[:, 1]

    def q_pos_slider_changed(self, value):
        self.demonstrate_button.config(state = "normal")
        # Convert the slider value to a suitable range for the x position
        x_position = float(value) / 10000

        # Calculate the joint angles that will result in the desired end-effector position
        # joint_angles = self.simulator.robot.inverse_kinematics(x_position, self.position_slider_y.get() / 10000)
        # joint_angles = self.simulator.robot.inverse_kinematics(x_position)
        joint_angles = x_position
        # Update the robot's joint angles
        self.simulator.robot.joint1_angle = joint_angles
        # Redraw the robot
        # self.simulator.draw_arm(self.velocity_slider_x.get() / 5000, self.velocity_slider_y.get() / 5000, self.force_slider_x.get() / 5000, self.force_slider_y.get() / 5000, ani=False, show_start_pt=self.show_strating_point)
        # self.simulator.draw_arm(self.velocity_slider_x.get() / 5000, self.velocity_slider_y.get() / 5000, self.force_slider_x.get() / 5000, self.force_slider_y.get() / 5000, ani=False)
        self.simulator.draw_arm(self.velocity_slider_q.get() / 5000, self.force_slider_q.get() / 5000, ani=False)
        
    
    def q_vel_slider_changed(self, value):
        self.demonstrate_button.config(state = "normal")
        # Convert the slider value to a suitable range for the x position
        x_velocity = float(value) / 5000
        # Keep the y position the same
        y_velocity = self.velocity_slider_q.get() / 5000
        # Calculate the joint angles that will result in the desired end-effector position
        joint_velocity = self.simulator.robot.get_joint_velocity_from_ee([x_velocity, y_velocity])
        # Update the robot's joint angles
        # self.simulator.robot.joint1_velocity, self.simulator.robot.joint2_velocity = joint_velocity[0], joint_velocity[1]
        self.simulator.robot.joint1_velocity = joint_velocity

        # Redraw the robot
        # self.simulator.draw_arm(self.velocity_slider_x.get() / 5000, self.velocity_slider_y.get() / 5000, self.force_slider_x.get() / 5000, self.force_slider_y.get() / 5000, ani=False, show_start_pt=self.show_strating_point)
        # self.simulator.draw_arm(self.velocity_slider_x.get() / 5000, self.velocity_slider_y.get() / 5000, self.force_slider_x.get() / 5000, self.force_slider_y.get() / 5000, ani=False)
        self.simulator.draw_arm(self.velocity_slider_q.get() / 5000, self.force_slider_q.get() / 5000, ani=False)
            
    def torque_slider_changed(self, value):
        self.demonstrate_button.config(state = "normal")
        self.force_slider_value_q.config(text=f"{float(value)/5000}")
        # self.simulator.draw_arm(self.velocity_slider_x.get() / 5000, self.velocity_slider_y.get() / 5000, self.force_slider_x.get() / 5000, self.force_slider_y.get() / 5000, ani=False, show_start_pt=self.show_strating_point)
        # self.simulator.draw_arm(self.joint1_angle, self.velocity_slider_q.get() / 5000, self.velocity_slider_y.get() / 5000, self.force_slider_q.get() / 5000, self.force_slider_q.get() / 5000, ani=False)
        self.simulator.draw_arm(self.velocity_slider_q.get() / 5000, self.force_slider_q.get() / 5000, ani=False)
            
    def demonstrate_btn_callback(self):
        # self.unlock_slider()
        if self.demo_cnt <= self.demo_num-1:
            
            self.phi_recording.append([self.position_slider_q.get()/10000, self.velocity_slider_q.get()/5000])
            
            if self.force_teach_flag:
                self.force_recording.append([self.force_slider_q.get()/5000, self.force_slider_q.get()/5000])

            print(f"{self.demo_cnt+1} demonstration completed.")
            print("phi input:   ", self.phi_recording[self.demo_cnt] + [1])
            print("force input: ", self.force_recording[self.demo_cnt])
            print("real force:  ", self.L_real @ np.array(self.phi_recording[self.demo_cnt] + [1]))

            self.demonstrate_button.config(state="disabled")
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
        elif len(self.phi_recording) !=0 and len(self.phi_recording) != 2:
            messagebox.showwarning('Warning', 'Incomplete demonstration.')
            return

        # phi = np.vstack((np.array(self.phi_recording).T, np.ones((1, 5))))
        phi = np.array(self.phi_recording).T

        if self.force_teach_flag == False: # force from calculated with noise
            force_real = self.L_real @ phi
            force_noi_diff = self.force_std_noise * np.random.randn(2, 5)
            force_with_noise = force_real + force_noi_diff
            force_given = force_with_noise
        elif self.force_teach_flag == True:
            force_teached = np.array(self.force_recording).T # np.shape (5,2)
            force_given = force_teached

        # calculate theta
        self.L_learner = self.simulator.learn(phi, force_given)
        print(f"Phase: {self.phase_cnt}: self.L_real = {self.L_real[0]}, {self.L_real[1]}")
        print(f"Phase: {self.phase_cnt}: self.L_learner = {self.L_learner[0]}, {self.L_learner[1]}")

        self.simulator.robot.set_L(self.L_learner)
        # show student's animation
        self.next_button.config(state='disabled')

        if self.phase_cnt in [1, 3, 5]:
            self.simulator.robot.joint1_angle  = self.simulator.robot.inverse_kinematics(0.8, 1.1)
        elif self.phase_cnt in [2, 4, 6]:
            self.simulator.robot.joint1_angle = self.simulator.robot.inverse_kinematics(0.2, 0.3)
        self.simulator.robot.joint1_velocity, self.simulator.robot.joint2_velocity = 0.0, 0.0
        for i in range(self.step):
            self.simulator.robot.trajectory.append(list(self.simulator.robot.get_end_effector_position()))
            self.simulator.robot.velocity.append(list(self.simulator.robot.get_end_effector_velocity()))
            # if True: # true = show student's animation, false = dont show
            if self.student_animation_show_flag:
                self.simulator.draw_arm()
            self.simulator.robot.update()
        self.student_trajectory = self.simulator.robot.trajectory

        # calculate error
        rmse = np.sqrt(np.mean((np.array(self.teacher_trajectory) - np.array(self.student_trajectory))**2))
        nmse = self.nmse_cal(self.L_real, self.L_learner)
        el2 = self.el2_cal(self.L_real, self.L_learner)
        print(f"rmse: {rmse}, nmse: {nmse}, el2: {el2}")
   
        # calculate score, but only show on 3th phase
        # guidance_score = self.score_calculator(phi, force_given)
        guidance_score, record_each_guidance = self.score_calculator(phi, force_given)
        print("score is: ", guidance_score)
        if self.phase_cnt == 3 and self.guidancetype:
            self.score_text.insert(tk.END, '\n {}.'.format(guidance_score)) # display score
            self.score_text.see(tk.END)

        # store data tocsv file
        self.store_data_tocsv(self.phase_cnt, self.trail_cnt, phi, force_given, self.L_learner, rmse, nmse, el2, guidance_score, record_each_guidance)
        self.store_data_tojson(phi, force_given, self.L_learner, rmse, nmse, el2, guidance_score, record_each_guidance)

        # reset robot
        self.simulator.reset_robot()
        self.reset_teaching_config()
        self.next_button.config(state='normal')

        # when phase_cnt = 6, exp. finished.
        # if self.phase_cnt >= 6 and self.trail_cnt == 3:
        if self.phase_cnt >= 6 and self.trail_cnt == self.max_trail_num:
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

    def store_data_tocsv(self, phase, trail, phi, force, theta_learned, rmse, nmse, el2, score, record_guidance):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        target_dir = os.path.join(os.path.join(current_dir, f'formal_exp_results'), f"formal_exp_id_{self.participant_number}")
        os.makedirs(target_dir, exist_ok=True)
        filename = os.path.join(target_dir, f"formal_exp_id_{self.participant_number}.csv")
        
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
            csvwriter.writerows([["nmse", f"{nmse}"]])
            csvwriter.writerows([["el2", f"{el2}"]])
            csvwriter.writerow([])

            csvwriter.writerows([["score", f"{score}"]])
            csvwriter.writerows([["d", f"{record_guidance[0]}", "Rho",f"{record_guidance[1]}",  "c",f"{record_guidance[2]}"]])
            csvwriter.writerow([])

            if self.phase_cnt == 3:
                csvwriter.writerows([["score", f"{score}"]])
                csvwriter.writerow([])

        print("Matrix saved to", filename)
        print("-----------------------------------------")  
    
    def store_data_tojson(self, phi, force_given, L, rmse, nmse, el2, score, record_guidance):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        target_dir = os.path.join(os.path.join(current_dir, f'formal_exp_results'), f"formal_exp_id_{self.participant_number}")
        # target_dir = os.path.join(current_dir, 'formal_exp_results', f"formal_exp_id_{self.participant_number}")
        filename = os.path.join(target_dir, f"formal_exp_id_{self.participant_number}.json")

        if os.path.exists(filename):
            with open(filename, 'r') as f:
                self.records = json.load(f)

        self.records['datas'].append({
            'phi': phi,
            'force': force_given,
            'theta': L,
            'rmse': rmse,
            'nmse': nmse,
            'el2': el2, 
            'score': score,
            'record each guidance': record_guidance
        })
        self.records['count'] += 1
        
        with open(filename, 'w') as f:
            json.dump(self.records, f, cls=NumpyEncoder)
         
    def SVD(self, phi):
        _, S, _ = np.linalg.svd(phi)
        return S
    
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

    def score_calculator(self, phi, force):
        norm_phi = normalize(phi, norm='l2', axis=0)
        force_e = np.array(force - self.L_real @ phi)

        d = np.sum(abs(self.student_learn(phi, force_e)))
        Rho = np.linalg.cond(phi)
        c = np.sum(np.linalg.norm(phi, axis=0))
        
        w1 = 100.0
        w2 = 0.01
        w3 = 0.001
        score = w1 * 1/d + w2 * 1/Rho + w3 * c
        # score = (w1 * 1/d + w2 * 1/Rho + w3 * c)
        

        return score, np.array([d, Rho, c])
        
    def student_learn(self, phi, force):
        lam = 1e-6
        I = np.identity(5)

        learned = (np.linalg.inv(phi @ phi.T + lam * I) @ phi @ (force.T)).T
        return learned



if __name__ == "__main__":
    # robot = SingleRobotArm(link1_length=1, link2_length=1, 
    #                     link1_mass=1, link2_mass=1, 
    #                     joint1_angle=0.0, joint2_angle=0.0, 
    #                     joint1_velocity=0.0, joint2_velocity=0.0, 
    #                     joint1_torque=0.0, joint2_torque=0.0,
    #                     time_step=0.05, g=9.81)
    
    robot = SingleRobotArm(
        link1_length=1, 
        link1_mass=1, 
        joint1_angle=0.0, 
        joint1_velocity=0.0,
        joint1_torque=0.0,
        time_step=0.05, 
        g=9.81)

    x0, y0 = 0.2, 1.5
    xt, yt = 1.5, 0.3

    app = RobotArmApp(robot=robot, demo_num=5, trial_num=3, force_teach=True, std=0.0, show_guidace_flag = True, student_animation = True)
    app.mainloop()
