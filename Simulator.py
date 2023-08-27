import os
import math
import datetime
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import tkinter.messagebox as messagebox
from sklearn.preprocessing import normalize
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from BasicRobotArm import BasicRobotArm
from utils import OrthogonalDemonstrationEvaluator

class RobotArmSimulator:
    def __init__(self, master, robot: BasicRobotArm, starting_point, target_point):
        self.master = master
        self.robot = robot

        self.init_x0, self.init_y0 = starting_point

        self.x0, self.y0 = starting_point
        self.xt, self.yt = target_point

        self.max_x_vel, self.max_y_vel = 0.5, 0.5 # m/s
        self.max_x_force, self.max_y_force = 1, 1 # N

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

        self.arm_line, = self.ax.plot([], [], 'o-', lw=3)
        self.starting_point, = self.ax.plot([], [], 'r*', ms=5)
        self.target_point, = self.ax.plot([], [], 'g*', ms=5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # initialize
        self.robot.joint1_angle, self.robot.joint2_angle = self.robot.inverse_kinematics(self.x0, self.y0)
        x, y = self.robot.forward_kinematics_from_states_2_joints([self.robot.joint1_angle, self.robot.joint2_angle])

        self.arm_line.set_data([0, x[0], x[1]], [0, y[0], y[1]])

        self.arrow_x = self.ax.annotate('', xy=(x[1] + 2, y[1]), xytext=(x[1],y[1]),arrowprops=dict(arrowstyle= '-|>',
                                                                                             color='blue', lw=0.5, ls='--'))
        self.arrow_y = self.ax.annotate('', xy=(x[1], y[1]+1), xytext=(x[1], y[1]),arrowprops=dict(arrowstyle= '-|>',
                                                                                             color='blue', lw=0.5, ls='--'))
        self.arrow = self.ax.annotate('', xy=(x[1] + 2, y[1] + 1), xytext=(x[1],y[1]),arrowprops=dict(arrowstyle= '-|>',
                                                                                             color='blue', lw=0.5, ls='solid'))
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

    def draw_arm(self, vel_len_x=0, vel_len_y=0, ani=True, show_arm=True, show_start_pt=True):
        x, y = self.robot.forward_kinematics_from_states_2_joints([self.robot.joint1_angle, self.robot.joint2_angle])

        self.arm_line.set_data([0, x[0], x[1]], [0, y[0], y[1]])

        self.starting_point.set_data([self.x0],[self.y0])
        self.target_point.set_data([self.xt], [self.yt])

        if not show_arm:
            self.arm_line.set_data([], [])

        if not show_start_pt:
            self.starting_point.set_data([], [])

        self.draw_arrow(vel_len_x, vel_len_y)
        self.fig.canvas.draw()
        if ani:
            self.fig.canvas.flush_events()

    def draw_arrow(self, vel_len_x, vel_len_y):
        x, y = self.robot.forward_kinematics_from_states_2_joints([self.robot.joint1_angle, self.robot.joint2_angle])

        self.arrow_x.remove()
        self.arrow_y.remove()
        self.arrow.remove()
        
        start_x = x[1]
        start_y = y[1]
        end_x = x[1] + vel_len_x * 0.8 # change arrow length
        end_y = y[1] + vel_len_y * 0.8

        self.arrow_x = self.ax.annotate('', xy=(end_x, start_y), xytext=(start_x,start_y), arrowprops=dict(arrowstyle= '-|>',
                                                                                             color='grey', lw=0.1, ls='--'))
        self.arrow_y = self.ax.annotate('', xy=(start_x, end_y), xytext=(start_x,start_y), arrowprops=dict(arrowstyle= '-|>',
                                                                                             color='grey', lw=0.1, ls='--'))
        self.arrow = self.ax.annotate('', xy=(end_x, end_y), xytext=(start_x,start_y), arrowprops=dict(arrowstyle= '-|>',
                                                                                             color='blue', lw=0.5, ls='solid'))

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

        self.demo_num = demo_num ## the number of demos

        self.mode = ''

        # the data need to be recorded
        self.phi_recording = []
        self.force_recording = []
        # self.norm_phis = []
        self.teacher_trajectory = []

        self.demo_cnt=0 # 5demos in one trail
        
        self.teaching_nums = 0

        self.participant_number = -1

        self.create_welcome_page()
        
        self.clock_show_flag = 1
        
        self.phase_cnt = 1 # only guidacne phase
        self.trail_cnt = 1
        self.max_trail_num = trial_num # trails in each phase, input

        self.phi1_list = []
        self.force1_real_list = []
        self.force1_noise_list = []
        self.l1_list = []
        self.phi2_list = []
        self.force2_real_list = []
        self.force2_noise_list = []
        self.l2_list = []

        self.evaluator = OrthogonalDemonstrationEvaluator()
        self.average_quality = -1

        self.force_teach_flag = force_teach
        self.std_noise = std
        
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
        self.demo_cnt = 0
        
        self.mode = ''

        self.phi_recording = []
        self.force_recording = []
        # self.norm_phis = []
        
        self.teaching_nums = 0
        self.L_learner = np.zeros((2, 5))
        self.teacher_trajectory = []

        self.evaluator._reset_evaluator()
        self.average_quality = -1

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

        # self.countdown_label = tk.Label(self.countdown_frame, font=(50), bg="light grey")
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
            print("Error: Invalid input. Please enter an integer.")

        self.create_exp_start_btn()
        # self.create_countdown_clock()
    
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

        # x slider label
        pos_slider_label_x = tk.Label(self.slide_frame,text='Position, x:   min', fg="blue")
        pos_slider_label_x.grid(row=0, column=0)
        self.position_slider_x = tk.Scale(self.slide_frame, from_= 0*10000, to=self.simulator.robot.x_max*10000, orient='horizontal', 
                                        command=self.x_pos_slider_changed, state='disabled', 
                                        length= 300, showvalue=0)
        self.position_slider_x.grid(row=0, column=1, pady =15)
        
        # x slider value
        self.pos_val_x = tk.StringVar()
        self.pos_val_x.set('max')
        self.pos_label_x = tk.Label(self.slide_frame, textvariable=self.pos_val_x, width=10, fg="blue")
        self.pos_label_x.grid(row=0, column=2)

        # y slider label
        pos_slider_label_y = tk.Label(self.slide_frame,text='Position, y:   min', fg="blue")
        pos_slider_label_y.grid(row=1, column=0)
        self.position_slider_y = tk.Scale(self.slide_frame, from_=0*10000, to=self.simulator.robot.y_max*10000, orient='horizontal', 
                                          command=self.y_pos_slider_changed, state='disabled', 
                                          length= 300, showvalue=0)
        self.position_slider_y.grid(row=1, column=1, pady=15)

        # y slider value
        self.pos_val_y = tk.StringVar()
        self.pos_val_y.set('max')
        self.pos_label_y = tk.Label(self.slide_frame, textvariable=self.pos_val_y, width=10, fg="blue")
        self.pos_label_y.grid(row=1, column=2)

        # xdot slider label
        vel_slider_label_x = tk.Label(self.slide_frame,text='Velocity, x:   min', fg="red")
        vel_slider_label_x.grid(row=2, column=0)
        self.velocity_slider_x = tk.Scale(self.slide_frame, from_=-self.simulator.max_x_vel*5000, to=self.simulator.max_x_vel*5000, orient='horizontal', 
                                          command=self.x_vel_slider_changed, state='disabled', 
                                          length= 300, showvalue=0)
        self.velocity_slider_x.grid(row=2, column=1, pady=15)
        # xdot slider value
        self.vel_val_x = tk.StringVar()
        self.vel_val_x.set("max")
        self.vel_label_x = tk.Label(self.slide_frame, textvariable=self.vel_val_x,  fg="red")
        self.vel_label_x.grid(row=2, column=2)

        # ydot slider label
        vel_slider_label_y = tk.Label(self.slide_frame,text='Velocity, y:   min', fg="red")
        vel_slider_label_y.grid(row=3, column=0, padx=20)
        self.velocity_slider_y = tk.Scale(self.slide_frame, from_=-self.simulator.max_y_vel*5000, to=self.simulator.max_y_vel*5000, orient='horizontal', 
                                          command=self.y_vel_slider_changed, state='disabled', 
                                          length= 300, showvalue=0)
        self.velocity_slider_y.grid(row=3, column=1, pady=15)
        # ydot slider value
        self.vel_val_y = tk.StringVar()
        self.vel_val_y.set("max")
        self.vel_label_y = tk.Label(self.slide_frame, textvariable=self.vel_val_y, fg="red")
        self.vel_label_y.grid(row=3, column=2)


        if self.force_teach_flag:
            # x force slider label
            force_slider_label_x = tk.Label(self.slide_frame,text='Force, x:   min', fg="green")
            force_slider_label_x.grid(row=4, column=0, padx=20)
            self.force_slider_x = tk.Scale(self.slide_frame, from_=-self.simulator.max_x_force*5000, to=self.simulator.max_x_force*5000, orient='horizontal', 
                                            command=self.x_force_slider_changed, state='disabled', 
                                            length= 300, showvalue=0)
            self.force_slider_x.grid(row=4, column=1, pady=15)
            # ydot slider value
            self.force_val_x = tk.StringVar()
            self.force_val_x.set("max")
            self.force_val_x = tk.Label(self.slide_frame, textvariable=self.force_val_x, fg="green")
            self.force_val_x.grid(row=4, column=2)
            self.force_slider_value_x = tk.Label(self.slide_frame,text='None', fg="green")
            self.force_slider_value_x.grid(row=4, column=3)
            self.force_slider_value_x.grid_forget()

            # y force slider label
            force_slider_label_y = tk.Label(self.slide_frame,text='Force, y:   min', fg="green")
            force_slider_label_y.grid(row=5, column=0, padx=20)
            self.force_slider_y = tk.Scale(self.slide_frame, from_=-self.simulator.max_y_force*5000, to=self.simulator.max_y_force*5000, orient='horizontal', 
                                            command=self.y_force_slider_changed, state='disabled', 
                                            length= 300, showvalue=0)
            self.force_slider_y.grid(row=5, column=1, pady=15)
            # ydot slider value
            self.force_val_y = tk.StringVar()
            self.force_val_y.set("max")
            self.force_val_y = tk.Label(self.slide_frame, textvariable=self.force_val_y, fg="green")
            self.force_val_y.grid(row=5, column=2)
            self.force_slider_value_y = tk.Label(self.slide_frame,text='None', fg="green")
            self.force_slider_value_y.grid(row=5, column=3)
            self.force_slider_value_y.grid_forget()
    
    def create_skill_mode_display(self):
        skill_frame = tk.Frame(self)
        skill_frame.grid(row=0, column=0, columnspan = 2)

        # self.text_display = tk.Label(skill_frame, text="Phase "+str(self.phase_cnt) +": Please teach Skill 1", font=("Arial", 20))
        # self.text_display = tk.Label(skill_frame, text="Please teach Skill 1 (Phase "+str(self.phase_cnt) , font=("Arial", 20))
        self.text_display = tk.Label(skill_frame, text="Phase11: Please teach Skill 1", font=("Arial", 20))
        print("Phase {}".format(self.phase_cnt))
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

    def create_score_indicator(self):
        self.score_frame = tk.Frame(self)
        self.score_frame.grid(row=3, column=0)

        guidance_label = tk.Label(self.score_frame,text='Score is :')
        guidance_label.grid(row=0, column=0)

        self.score_text = tk.Text(self.score_frame, width=30, height=1, wrap=tk.WORD)
        self.score_text.grid(row=0, column=1, padx=10)  # Adjusted pady
        
        # self.score_text.insert(tk.END, '\n Waiting.')
        self.score_text.insert(tk.END, '\n          ')
        self.score_text.see(tk.END)

    def mode_selection(self, skill_var):
        self.reset_teaching_config()
        self.unlock_slider()
        self.demonstrate_button.config(state='normal')

        self.position_slider_x.set(0.0)
        self.position_slider_y.set(0.0)
        self.velocity_slider_x.set(0.0)
        self.velocity_slider_y.set(0.0)

        if skill_var == 1:
            self.mode = 'S1NG'
            self.L_set(1)
        elif skill_var == 3:
            self.mode = 'S1G'
            self.L_set(1)
        elif skill_var == 5:
            self.mode = 'S1NG'
            self.L_set(1)
        elif skill_var == 2:
            self.mode = 'S2NG'
            self.L_set(2)
        elif skill_var == 4:
            self.mode = 'S2NG'
            self.L_set(2)
            self.score_frame.destroy()
        elif skill_var == 6:
            self.mode = 'S2NG'
            self.L_set(2)        

        self.simulator.robot.set_L(self.L)
        
        self.demo()
        self.simulator.draw_arm(ani=False, show_arm=False, show_start_pt=True)

        if self.mode == 'S1NG' or self.mode == 'S2NG':
            pass
        elif self.mode == 'S1G':
            self.create_score_indicator()
            self.score_text.insert(tk.END, '\n          ')
            self.score_text.see(tk.END)

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

    def reset_force_slider(self):
        if self.force_teach_flag:
            self.force_slider_x.set(0)
            self.force_slider_y.set(0)
    
    def demo(self):
        if (self.target_point[0] - 0) ** 2 + (self.target_point[1] - 0) ** 2 <= 2 ** 2:
            print("self.L", self.L)
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
        self.simulator.draw_arm(self.velocity_slider_x.get() / 5000, self.velocity_slider_y.get() / 5000, ani=False, show_start_pt=False)

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
        self.simulator.draw_arm(self.velocity_slider_x.get() / 5000, self.velocity_slider_y.get() / 5000, ani=False, show_start_pt=False)

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
        self.simulator.draw_arm(self.velocity_slider_x.get() / 5000, self.velocity_slider_y.get() / 5000, ani=False, show_start_pt=False)

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
        self.simulator.draw_arm(self.velocity_slider_x.get() / 5000, self.velocity_slider_y.get() / 5000, ani=False, show_start_pt=False)

    def x_force_slider_changed(self, value):
        self.force_slider_value_x.config(text=f"{float(value)/5000}")

    def y_force_slider_changed(self, value):
        self.force_slider_value_y.config(text=f"{float(value)/5000}")

    def next_phase_btn_callback(self):
        if self.trail_cnt < self.max_trail_num:
            self.trail_cnt += 1
        else:
            if self.phase_cnt <= 6:
                self.phase_cnt += 1
            self.trail_cnt = 1
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
        print("Phase {}".format(self.phase_cnt))
        
        
        self.reset_canvas_config()
        self.mode_selection(skill_var=self.phase_cnt)

        self.unlock_slider()
        self.reset_force_slider()

    def demonstrate_btn_callback(self):
        # self.unlock_slider()
        if self.demo_cnt <= self.demo_num-1:
            self.phi_each = [self.position_slider_x.get()/10000, self.position_slider_y.get()/10000, self.velocity_slider_x.get()/5000, self.velocity_slider_y.get()/5000]
            self.phi_recording.append(self.phi_each)
            
            if self.force_teach_flag:
                self.force_recording.append([self.force_slider_x.get()/5000, self.force_slider_y.get()/5000])

            self.demo_cnt += 1
            
            self.phi_each = [self.position_slider_x.get()/10000, self.position_slider_y.get()/10000, self.velocity_slider_x.get()/5000, self.velocity_slider_y.get()/5000, 1]

            if self.demo_cnt == self.demo_num:
                self.average_quality, overall_feedbacks = self.evaluator.evaluate_demonstrations()
                # print(f"Average Quality Score: {self.average_quality}")

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

        phi = np.vstack((np.array(self.phi_recording).T, np.ones((1, 5))))

        if self.force_teach_flag == False:
            force_real = self.L @ phi
            force_noi_diff = self.std_noise * np.random.randn(2, 5)
            force_noise = force_real + force_noi_diff
            self.L_learner = self.simulator.learn(phi, force_noise)
        else:
            force_teached = np.array(self.force_recording).T # np.shape (5,2)
            self.L_learner = self.simulator.learn(phi, force_teached)

        # score1 = self.score1(phi, force_teached)
        if self.force_teach_flag:
            score1 = self.score1(phi, force_teached)
            print("score1:", score1)
        else:
            score1 = self.score1(phi, force_noise)
            print("score1:", score1)

        self.simulator.robot.set_L(self.L_learner)
        if '1' in self.mode:
            self.simulator.robot.joint1_angle, self.simulator.robot.joint2_angle = self.simulator.robot.inverse_kinematics(0.8, 1.1)
        elif '2' in self.mode:
            self.simulator.robot.joint1_angle, self.simulator.robot.joint2_angle = self.simulator.robot.inverse_kinematics(0.2, 0.3)

        self.simulator.robot.joint1_velocity, self.simulator.robot.joint2_velocity = 0.0, 0.0

        self.next_button.config(state='disabled')
        for i in range(self.step):
            self.simulator.robot.trajectory.append(list(self.simulator.robot.get_end_effector_position()))
            self.simulator.robot.velocity.append(list(self.simulator.robot.get_end_effector_velocity()))
            self.simulator.draw_arm()
            self.simulator.robot.update_rk4()

        self.student_trajectory = self.simulator.robot.trajectory

        rmse = np.sqrt(np.mean((np.array(self.teacher_trajectory) - np.array(self.student_trajectory))**2))
        nmse = np.linalg.norm(self.L_learner - self.L)
        el2 = np.linalg.norm(self.L_learner - self.L)

        print("learned L",self.L_learner)
        print("rmse", rmse)
        print("el2", el2)

        # record score and rmse
        file_path = f"pilot_forceteached_{self.participant_number}_exp.txt"
        
        with open(file_path, 'a') as file:
            file.write(f"-------USER ID: {self.participant_number}, Phase {self.phase_cnt}-------\n")
            file.write(f"Skill Mode: {self.mode}\n")
            file.write(f'RMSE: {rmse}\n')
            file.write(f'NMSE: {nmse}\n')
            file.write(f'El2: {el2}\n')
            file.write(f"L desired:\n{self.L}\n")
            file.write(f"L learned:\n{self.L_learner}\n")
            file.write(f"phi:\n{phi}\n")
            # file.write(f"min sigma: {min_sigma_record}\n")
            file.write(f'average quality: {self.average_quality}\n')
            if self.force_teach_flag:
                file.write(f"force_teached:\n{force_teached}\n")
            else:
                file.write(f"force_real:\n{force_real}\n")
                file.write(f"force_noise:\n{force_noise}\n")
                file.write(f"force_noi_diff:\n{force_noi_diff}\n")

        if self.mode == 'S1G' or self.mode == 'S2G':
            fit_score_s1 = self.fit_score_s1(np.min(np.linalg.svd(normalize(phi, norm='l2', axis=0))[1]))

            guidance_score = self.score(phi)
            self.score_text.insert(tk.END, '\n {}.'.format(guidance_score))
            self.score_text.see(tk.END)

            with open(file_path, 'a') as file:
                file.write(f"Score: {guidance_score}\n\n\n")
        elif self.mode == 'S1NG' or self.mode  == 'S2NG':
            with open(file_path, 'a') as file:
                file.write(f"No Score.\n\n\n")

        if self.mode == 'S1G' or self.mode == 'S1NG':
            self.phi1_list.append(phi)
            if self.force_teach_flag:
                self.l1_list.append(force_teached)
            else:
                self.force1_real_list.append(force_real)
                self.force1_noise_list.append(force_noise)
            
        elif self.mode == 'S2G' or self.mode  == 'S2NG':
            self.phi2_list.append(phi)        
            if self.force_teach_flag:
                self.l2_list.append(force_teached)
            else:
                self.force2_real_list.append(force_real)
                self.force2_noise_list.append(force_noise)

        self.simulator.reset_robot()
        self.reset_teaching_config()
        
        self.next_button.config(state='normal')

        if self.phase_cnt == 6 and self.trail_cnt == 3:
            print("Finished!")
            messagebox.showinfo("Information", "Experiment Finished.")
            self.store_date_tonpy()
            self.quit()

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

    def store_date_tonpy(self):
        script_folder = os.path.dirname(os.path.realpath(__file__))
        subfolder_path = os.path.join(script_folder, f'pilot_forceteached_exp_{self.participant_number}')
        os.makedirs(subfolder_path, exist_ok=True)  # creates the subfolder if it does not exist

        phi1_file_path = os.path.join(subfolder_path, f'pilot_{self.participant_number}_skill1_phi.npy')
        if self.force_teach_flag == False:
            force1_real_file_path = os.path.join(subfolder_path, f'pilot_{self.participant_number}_skill1_force_real.npy')
            force1_noise_file_path = os.path.join(subfolder_path, f'pilot_{self.participant_number}_skill1_force_noise.npy')
        else:
            force1_teached_file_path = os.path.join(subfolder_path, f'pilot_{self.participant_number}_skill1_teached_real.npy')
        L1_file_path = os.path.join(subfolder_path, f'pilot_{self.participant_number}_skill1_L.npy')

        phi2_file_path = os.path.join(subfolder_path, f'pilot_{self.participant_number}_skill2_phi.npy')
        if self.force_teach_flag == False:
            force2_real_file_path = os.path.join(subfolder_path, f'pilot_{self.participant_number}_skill2_force_real.npy')
            force2_noise_file_path = os.path.join(subfolder_path, f'pilot_{self.participant_number}_skill2_force_noise.npy')
        else:
            force2_teached_file_path = os.path.join(subfolder_path, f'pilot_{self.participant_number}_skill2_teached_real.npy')
        L2_file_path = os.path.join(subfolder_path, f'pilot_{self.participant_number}_skill2_L.npy')
        
        np.save(phi1_file_path, self.phi1_list)
        if self.force_teach_flag == False:
            np.save(force1_real_file_path, self.force1_real_list)
            np.save(force1_noise_file_path, self.force1_noise_list)
        else:
            np.save(force1_teached_file_path, self.force_recording)
        np.save(L1_file_path, self.l1_list)
        
        np.save(phi2_file_path,  self.phi2_list)
        if self.force_teach_flag == False:
            np.save(force2_real_file_path, self.force2_real_list)
            np.save(force2_noise_file_path, self.force2_noise_list)
        else:
            np.save(force1_teached_file_path, self.force_recording)
        np.save(L2_file_path, self.l2_list)

    def plot_dynamic_motion(self):
        # Create a figure with 4 subplots
        trajectory = np.array(self.simulator.robot.trajectory)
        velocity = np.array(self.simulator.robot.velocity)

        fig, axs = plt.subplots(4, 2)
        fig.suptitle("{}th Teaching Dynamics Motion \n\n from StartingPoint:{} to TargetPoint:{} \n\n L = {}\n         {}".format(self.teaching_nums, self.starting_point, self.target_point, self.robot.L[0], self.robot.L[1]))
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