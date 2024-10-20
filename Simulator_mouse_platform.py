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

from matplotlib.legend_handler import HandlerLine2D
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerPatch

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

class ProgressBar(tk.Frame):
    def __init__(self, parent, steps, start_step=0, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.configure(background='white')
        self.steps = steps
        self.current_step = start_step
        self.canvas = tk.Canvas(self, height=50, width=600, bg='white')
        self.canvas.pack(fill='both', expand=True)
        self.draw_progress_bar()
        
    def draw_progress_bar(self):
        self.canvas.delete("all")  # Clear the canvas
        circle_radius = 10
        line_length = (self.canvas.winfo_reqwidth() - (self.steps * circle_radius * 2) - 20) // (self.steps - 1)
        y_length = (self.canvas.winfo_reqheight() - circle_radius * 2 - 20) // 2

        # Draw circles and lines
        for i in range(self.steps):
            x = (circle_radius * 2) * i + line_length * i + circle_radius + 10
            self.canvas.create_oval(x - circle_radius, circle_radius + y_length, 
                                    x + circle_radius, circle_radius * 3 + y_length, 
                                    fill='red' if i < self.current_step else 'white',
                                    outline='black')

            if i < self.steps - 1:
                self.canvas.create_line(x + circle_radius, circle_radius * 2 + y_length,
                                        x + circle_radius + line_length, circle_radius * 2 + y_length,
                                        fill='black' if i < self.current_step - 1 else 'black',
                                        width=2)

    def set_step(self, step):
        """Set the progress to the specified step."""
        if 0 <= step < self.steps:
            self.current_step = step
            self.draw_progress_bar()
        else:
            raise ValueError("Step out of range")

    def advance_progress(self):
        """Advance the progress by one step."""
        if self.current_step < self.steps:
            self.current_step += 1
            self.draw_progress_bar()

    def reset(self, num):
        self.steps = num
        self.current_step = 0
        self.draw_progress_bar()

class ArrowDrawer:
    def __init__(self, ax, given_states, optimal_force, show_optiaml_force):
        self.all_drawn_elements = []
        self.keep_arrows = []
        self.ax = ax

        self.given_states = given_states

        self.start_positions = (self.given_states[:2].T).tolist()
        self.given_velocity = (self.given_states[2:4].T).tolist()
        self.optimal_force = optimal_force.T #nparray

        self.start_vel_markers = []
        self.force_markers = []
        self.start_markers = []
        self.target_circles = []
        self.optimal_arrows = []
        self.arrow = None
        self.arrow_counter = 0
        self.can_draw = True
        self.slider_ratio = 0.2
        # self.arrow_length_limit = 0.66
        # self.arrow_length_limit = self.slider_ratio * 2.2 * np.sqrt(2)
        self.arrow_length_limit = self.slider_ratio * 3.0

        self.show_optiaml_force = show_optiaml_force

        self.pos_color = 'orange'
        self.vel_color = 'blue'
        self.force_color = 'green'
        self.optimal_force_color = 'gray'
        self.circle_color = 'gray'

        self.optimal_force_color = 'lightgreen'
        self.circle_color = 'lightgreen'


        self.arrow_x_len = None
        self.arrow_y_len = None

        self.x_input_value = []
        self.y_input_value = []

        x, y = self.start_positions[0]
        marker = self.ax.plot(x, y, marker='o', markersize=5, color=self.pos_color)[0]
        self.start_markers.append(marker)

        xvel, yvel = self.given_velocity[0]
        marker_vel = self.ax.arrow(x, y,
                                   xvel*self.slider_ratio, yvel*self.slider_ratio,
                                   width=0.01, length_includes_head=True,
                                   color=self.vel_color)
        print("xvel, yvel", xvel, yvel)
        self.start_vel_markers.append(marker_vel) 

        for pos in self.start_positions:
            circle = plt.Circle(pos, self.arrow_length_limit, color=self.optimal_force_color, fill=False, linestyle='dashed')
            self.ax.add_artist(circle)
            circle.set_visible(False)
            self.target_circles.append(circle)

        self.target_circles[self.arrow_counter].set_visible(True)


    def clear_all_elements(self):
        # Remove all drawn elements
        for element in self.all_drawn_elements:
            element.remove()
        # Remove arrows that are kept
        for arrow in self.keep_arrows:
            arrow.remove()
        # Remove start markers
        for marker in self.start_markers:
            marker.remove()
        # Remove velocity markers
        for marker in self.start_vel_markers:
            marker.remove()
        # Remove target circles
        for circle in self.target_circles:
            circle.remove()  # This removes the circle from the axis
        
        if self.show_optiaml_force:
            for optf in self.optimal_arrows:
                optf.remove()
        # Redraw the figure canvas to reflect the removals and resets
        self.ax.figure.canvas.draw()

    def reinit(self):
        self.all_drawn_elements = []
        self.keep_arrows = []

        self.start_positions = (self.given_states[:2].T).tolist()
        self.given_velocity = (self.given_states[2:4].T).tolist()

        self.start_vel_markers = []
        self.force_markers = []
        self.start_markers = []
        self.target_circles = []
        self.optimal_arrows = []
        self.arrow = None
        self.arrow_counter = 0
        self.can_draw = True

        self.arrow_x_len = None
        self.arrow_y_len = None

        self.x_input_value = []
        self.y_input_value = []

        x, y = self.start_positions[0]
        marker = self.ax.plot(x, y, marker='o', markersize=5, color=self.pos_color)[0]
        self.start_markers.append(marker)

        xvel, yvel = self.given_velocity[0]
        marker_vel = self.ax.arrow(x, y,
                                   xvel*self.slider_ratio, yvel*self.slider_ratio,
                                   width=0.01, length_includes_head=True,
                                   color=self.vel_color)
        print("xvel, yvel", xvel, yvel)
        self.start_vel_markers.append(marker_vel) 

        for pos in self.start_positions:
            circle = plt.Circle(pos, self.arrow_length_limit, color=self.circle_color, fill=False, linestyle='dashed')
            self.ax.add_artist(circle)
            circle.set_visible(False)
            self.target_circles.append(circle)

        self.target_circles[self.arrow_counter].set_visible(True)

    def add_element(self, element):
        """Add a drawn element to the list for future reference."""
        self.all_drawn_elements.append(element)

    def update_arrow(self, event):
        if event.button == 3 and self.arrow_counter < len(self.start_positions) and self.can_draw:  # 右键拖动鼠标更新箭头位置
            if event.xdata == None or event.ydata == None:
                return
            if len(self.all_drawn_elements) != 0:
                for a in self.all_drawn_elements:
                    a.remove()
                self.all_drawn_elements = []

            # 计算箭头的长度
            length = np.sqrt((event.xdata - self.start_positions[self.arrow_counter][0])**2 +
                             (event.ydata - self.start_positions[self.arrow_counter][1])**2)


            # # 如果箭头长度超过0.66，则将其截断为长度为0.66的箭头
            if 0 < length <= self.arrow_length_limit:
                self.arrow_x_len = (event.xdata - self.start_positions[self.arrow_counter][0])
                self.arrow_y_len = (event.ydata - self.start_positions[self.arrow_counter][1])
            elif length > self.arrow_length_limit:
                self.arrow_x_len = self.arrow_length_limit * (event.xdata - self.start_positions[self.arrow_counter][0]) / length
                self.arrow_y_len = self.arrow_length_limit * (event.ydata - self.start_positions[self.arrow_counter][1]) / length
                
            # 绘制箭头
            self.arrow = self.ax.arrow(self.start_positions[self.arrow_counter][0],
                                       self.start_positions[self.arrow_counter][1],
                                       self.arrow_x_len, self.arrow_y_len,
                                       width=0.01, length_includes_head=True,
                                       color=self.force_color)
            self.add_element(self.arrow)
            
            self.ax.figure.canvas.draw()

    def confirm_arrow(self):
        if self.arrow_counter < len(self.start_positions) and len(self.all_drawn_elements) != 0:
            self.target_circles[self.arrow_counter].set_visible(False)

            input_x = round(self.arrow_x_len / self.slider_ratio, 2) * 2
            input_y = round(self.arrow_y_len / self.slider_ratio, 2) * 2
            print("arrow input, x and y:", input_x, input_y)

            self.x_input_value.append(input_x)
            self.y_input_value.append(input_y)

            # self.x_input_value.append(self.arrow_x_len / self.slider_ratio)
            # self.y_input_value.append(self.arrow_y_len / self.slider_ratio)

            print("self.arrow_x_len, self.arrow_y_len : ", self.x_input_value, self.y_input_value)

            self.can_draw = False
            self.keep_arrows.append(self.all_drawn_elements[-1])
            self.all_drawn_elements = []
            
            if self.show_optiaml_force:
                self.optimal_arrow = self.ax.arrow(self.start_positions[self.arrow_counter][0],
                                        self.start_positions[self.arrow_counter][1],
                                        self.optimal_force[self.arrow_counter][0]*self.slider_ratio/2,
                                        self.optimal_force[self.arrow_counter][1]*self.slider_ratio/2,
                                        width=0.01, length_includes_head=True,
                                        color='lightgreen')
                self.ax.figure.canvas.draw()
                self.optimal_arrows.append(self.optimal_arrow)
            return True
        else:
            return False

    def prepare_next(self):
        if self.arrow_counter < len(self.start_positions) - 1:
            self.can_draw = True
            self.arrow_counter += 1

            self.target_circles[self.arrow_counter].set_visible(True)
            
            x, y = self.start_positions[self.arrow_counter]
            marker = self.ax.plot(x, y, marker='o', markersize=5, color=self.pos_color)[0]
            self.start_markers.append(marker)

            xvel, yvel = self.given_velocity[self.arrow_counter]
            marker_vel = self.ax.arrow(x, y, xvel*self.slider_ratio, yvel*self.slider_ratio, width=0.01, length_includes_head=True, color=self.vel_color)
            self.start_vel_markers.append(marker_vel)

            self.ax.figure.canvas.draw()
        else:
            pass

class HandlerCircle(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):
        # This method overrides the default artist creation for the patch handler.
        # Adjust the circle's diameter based on the legend's fontsize to ensure consistent sizing.
        diameter = fontsize * 0.75
        # Create the circle with the correct size and alignment.
        circle = mpatches.Circle(xy=(width / 2 - xdescent, height / 2 - ydescent), 
                                 radius=diameter / 2,
                                 facecolor=orig_handle.get_facecolor(),
                                 edgecolor=orig_handle.get_edgecolor(),
                                 linestyle=orig_handle.get_linestyle(),
                                 linewidth=orig_handle.get_linewidth(),
                                 transform=trans)
        return [circle]
    
class RobotArmSimulator:
    def __init__(self, master, robot: BasicRobotArm):
        self.master = master
        self.robot = robot
        self.arm_animation = False
        self.arm_animation = True

        self.init_x0, self.init_y0 = 0.2111, 0.3111 # 0.2, 0.5

        #setting of velociy, force
        self.max_x_vel, self.max_y_vel = 1, 1
        self.max_x_force, self.max_y_force = 2.2, 2.2

        self.positions = []
        self.velocities = []
        self.accelerations = []
        self.joint_angles = []
        self.joint_velocities = []
        self.joint_accelerations = []

        self.fig = Figure(figsize=(6, 6), dpi=100)
        # self.fig = Figure(figsize=(2, 2), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
        
        self.ax.set_xlim(-0.3, self.robot.x_max + 0.3)
        self.ax.set_ylim(-0.6, self.robot.y_max + 0.3)
        self.ax.set_aspect('equal', 'box')

        # 画网格代码
        self.ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        self.ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
        self.ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
        self.ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))
        self.ax.grid(which='minor', color='gray', linestyle=':')
        self.ax.grid(which='major', color='gray', linestyle=':')

        self.ax.xaxis.set_ticks([])  # 隐藏x轴刻度
        self.ax.yaxis.set_ticks([])
        

        self.arrow_head_width = 0.07
        self.arrow_head_length = 0.015

        # 画坐标轴 参数依次是起始点坐标、箭头的水平长度、箭头头部的宽度和长度，颜色等
        # self.ax.arrow(-0.3, 0, 2.5, 0, head_width=self.arrow_head_width, head_length=self.arrow_head_length, color='grey', overhang=0.2)  
        # self.ax.arrow(0, -0.3, 0, 2.5, head_width=self.arrow_head_width, head_length=self.arrow_head_length, color='grey', overhang=0.2)
        
        # 画坐标轴
        self.ax.plot([0, 0], [0, 2], color='grey', linestyle='-')
        self.ax.plot([0, 2], [0, 0], color='grey', linestyle='-')
        self.ax.plot([2, 2], [0, 2], color='grey')
        self.ax.plot([0, 2], [2, 2], color='grey')

        self.starting_point, = self.ax.plot([], [], 'g*', ms=10)
        self.target_point, = self.ax.plot([], [], 'r*', ms=10)
        self.target_line, = self.ax.plot([], [], color='red', linestyle='--', alpha = 0.6,)
        # self.target_line.set_data([0.8, 0.8], [-0.1, 2.1])


        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # initialize starting point
        self.robot.joint1_angle, self.robot.joint2_angle = self.robot.inverse_kinematics(self.init_x0, self.init_y0)
        x, y = self.robot.forward_kinematics_from_states_2_joints([self.robot.joint1_angle, self.robot.joint2_angle])

        if self.arm_animation == False:
            self.arm_line, = self.ax.plot([], [], 'o-', lw=3, alpha = 0.6, label='Robot Arm') # robot arm toumingdu
            self.arm_line.set_data([0, x[0], x[1]], [0, y[0], y[1]])

        

        # self.arrow = self.ax.annotate('', xy=(x[1] + 2, y[1] + 1), xytext=(x[1],y[1]),arrowprops=dict(arrowstyle= '-|>',
        #                                                                                      color='blue', lw=0.5, ls='solid'))
        # self.force_arrow = self.ax.annotate('', xy=(x[1] + 2, y[1] + 1), xytext=(x[1],y[1]),arrowprops=dict(arrowstyle= '-|>',
        #                                                                                      color='red', lw=0.5, ls='solid'))


        # self.arrow = self.ax.arrow(x[1],y[1],x[1] + 0.1, y[1] + 1, length_includes_head=True, head_width=0.05, head_length=0.1, fc='blue', ec='blue')
        # self.force_arrow = self.ax.arrow(x[1],y[1],x[1] + 0.5, y[1] + 2, length_includes_head=True, head_width=0.05, head_length=0.1, fc='red', ec='red')

        # self.arrow_x = self.ax.arrow(x[1],y[1],x[1] + 0.1, 0, length_includes_head=True, head_width=self.arrow_head_width, head_length=self.arrow_head_length, fc='blue', ec='blue')
        # self.arrow_y = self.ax.arrow(x[1],y[1],0, y[1] + 1, length_includes_head=True, head_width=self.arrow_head_width, head_length=self.arrow_head_length, fc='blue', ec='blue')
        # self.force_arrow_x = self.ax.arrow(x[1],y[1],x[1] + 0.5, 0, length_includes_head=True, head_width=self.arrow_head_width, head_length=self.arrow_head_length, fc='red', ec='red')
        # self.force_arrow_y = self.ax.arrow(x[1],y[1],0, y[1] + 2, length_includes_head=True, head_width=self.arrow_head_width, head_length=self.arrow_head_length, fc='red', ec='red')
        
        # self.real_force_arrow_x = self.ax.arrow(x[1],y[1],x[1] + 0.5, 0, length_includes_head=True, head_width=self.arrow_head_width, head_length=self.arrow_head_length, fc='grey', ec='grey')
        # self.real_force_arrow_y = self.ax.arrow(x[1],y[1],0, y[1] + 0.5, length_includes_head=True, head_width=self.arrow_head_width, head_length=self.arrow_head_length, fc='grey', ec='grey')

    
        self.draw_arm(ani=False)
        arrow1 = mlines.Line2D([], [], color='blue', marker='>', linestyle='-', linewidth=2, markersize=8, label='Velocity')
        arrow2 = mlines.Line2D([], [], color='green', marker='>', linestyle='-', linewidth=2, markersize=8, label='Your Force')
        dummy_circle = mpatches.Circle((0, 0), 1, facecolor='none', edgecolor='lightgreen', linestyle='dashed', linewidth=0.5, label='Force Max Value')

        # Creating the legend
        # fig, ax = plt.subplots()
        self.fig.legend(handles=[arrow1, arrow2, dummy_circle],
                bbox_to_anchor=(0.905, 0.168),
                handler_map={mpatches.Circle: HandlerCircle()},
                bbox_transform=self.fig.transFigure)

    def mouse_input(self, given_states=[], optimal_force=[]):
        start_positions = given_states
        
        if start_positions != []:
            # self.arrow_drawer = ArrowDrawer(self.ax, start_positions)
            self.arrow_drawer = ArrowDrawer(self.ax, given_states, optimal_force, False)
            self.canvas.mpl_connect('motion_notify_event', self.arrow_drawer.update_arrow)
            # self.canvas.mpl_connect('button_release_event', self.arrow_drawer.mouse_release)
            # 绑定按键事件
            self.master.bind("<Key>", self.key_pressed)

            # 每次鼠标释放事件都调用on_button_release函数
            self.canvas.mpl_connect('button_release_event', self.on_button_release)
        else:
            pass

    def key_pressed(self, event):
        print("Key pressed:", event.char)

    # 在ArrowDrawer类定义之外定义on_button_release函数
    def on_button_release(self, event):
        dx_values = self.arrow_drawer.x_input_value
        dy_values = self.arrow_drawer.y_input_value

        return dx_values, dy_values

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
    
    def draw_arm(self, vel_len_x=0, vel_len_y=0, force_len_x=0, force_len_y=0, ani=True, show_arm=True, real_force_len_x=0, real_force_len_y=0):
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

        # self.draw_arrow(vel_len_x, vel_len_y, force_len_x, force_len_y)
        # self.draw_real_force_arrow(real_force_len_x, real_force_len_y)
        self.fig.canvas.draw()
        if ani:
            self.fig.canvas.flush_events()

    def draw_arrow(self, vel_len_x, vel_len_y, force_len_x, force_len_y):
        x, y = self.robot.forward_kinematics_from_states_2_joints([self.robot.joint1_angle, self.robot.joint2_angle])

        # self.arrow.remove()
        # self.force_arrow.remove()
        
        self.arrow_x.remove()
        self.arrow_y.remove()
        self.force_arrow_x.remove()
        self.force_arrow_y.remove()

        start_x = x[1]
        start_y = y[1]

        ratio = 0.2
    
        vel_end_x = vel_len_x * ratio
        vel_end_y = vel_len_y * ratio
        force_end_x = force_len_x * ratio
        force_end_y = force_len_y * ratio

        self.arrow_x = self.ax.arrow(start_x,start_y, vel_end_x, 0, length_includes_head=True, head_width=self.arrow_head_width, head_length=self.arrow_head_length, fc='blue', ec='blue')
        self.arrow_y = self.ax.arrow(start_x,start_y, 0, vel_end_y, length_includes_head=True, head_width=self.arrow_head_width, head_length=self.arrow_head_length, fc='blue', ec='blue')

        self.force_arrow_x = self.ax.arrow(start_x,start_y, force_end_x, 0, length_includes_head=True, head_width=self.arrow_head_width, head_length=self.arrow_head_length, fc='red', ec='red')
        self.force_arrow_y = self.ax.arrow(start_x,start_y, 0, force_end_y, length_includes_head=True, head_width=self.arrow_head_width, head_length=self.arrow_head_length, fc='red', ec='red')

    def draw_real_force_arrow(self, real_force_len_x, real_force_len_y):
        x, y = self.robot.forward_kinematics_from_states_2_joints([self.robot.joint1_angle, self.robot.joint2_angle])
        
        self.real_force_arrow_x.remove()
        self.real_force_arrow_y.remove()
        
        start_x = x[1]
        start_y = y[1]
        ratio = 0.2

        real_force_end_x = real_force_len_x * ratio
        real_force_end_y = real_force_len_y * ratio

        self.real_force_arrow_x = self.ax.arrow(start_x,start_y, real_force_end_x, 0, length_includes_head=True, head_width=self.arrow_head_width, head_length=self.arrow_head_length, fc='grey', ec='grey')
        self.real_force_arrow_y = self.ax.arrow(start_x,start_y, 0, real_force_end_y, length_includes_head=True, head_width=self.arrow_head_width, head_length=self.arrow_head_length, fc='grey', ec='grey')

    def learn(self, phi, force, demo=5, lam=1e-4):
        # lam = 1e-4
        lam = lam
        I = np.identity(demo)

        learned_thetea = (np.linalg.inv(phi @ phi.T + lam * I) @ phi @ (force.T)).T
        return learned_thetea
    
    def learn_lambda6(self, phi, force, demo=5):
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
    def __init__(self, robot: BasicRobotArm, demo_num, trial_num, pilot_num, force_teach, state_teach, plot_result_figure_flag, std, show_guidace_flag, student_animation, administrator_mode, visual_guidacne):
        super().__init__()
        self.visual_guidance = visual_guidacne
        self.configure(background='white')
        self.geometry("605x800") 

        self.robot = robot
        self.administrator_mode = administrator_mode
        
        self.slider_scale = 10000
        self.slider_scale = 100  # 小数点后2位

        
        # self.target_point = [1.237, 0.592]
        self.target_point = [1.22857732, 0.58413413]
        self.real_theta = np.zeros((2, 5))
        self.learned_theta = np.zeros((2, 5))
 
        # self.theta_1 = np.array([[-1, 0, -1, 0, 0.8],
        #                         [0, -1, 0, -1, 1.2]])   
        # self.theta_2 = np.array([[-1, 0, -1, 0, 0.8],
        #                         [0, 0, 0, -1, 0]])
        

        self.theta_1 = np.array([[-1, 0, -3, 0, 0.8],
                                [0, -1, 0, -3, 1.2]])   
        self.theta_2 = np.array([[-3, 0, -1, 0, 0.8],
                                [0, 0, 0, -1, 0]])
        


        self.step = 0        

        # the data need to be recorded
        self.phi_recording = []
        self.force_recording = []
        self.teacher_trajectory = []

        self.user_number = -1
        self.create_page_1()
        self.clock_show_flag = 1
        
        # self.demo_cnt= 0                # 5 demos in one trail
        # self.max_demo_num = demo_num        # the number of demos
        # self.max_demo_num = 10
        # self.trail_cnt = 1              # 3 trials in one phase
        # self.max_trail_num = trial_num  # 3 trails in each phase, input
        # self.phase_cnt =  1             # 6 phase in total
        # self.max_phase_num = 3
        # self.teaching_phase_num = 2     # phase 2, teaching phase

        # self.phases = [
        #     {'trials': 1, 'demos_per_trial': 6, 'state': '11111', 'current_trial': 0, 'current_demo': 0},
        #     {'trials': 8, 'demos_per_trial': 5, 'state': '00000', 'current_trial': 0, 'current_demo': 0},
        #     {'trials': 1, 'demos_per_trial': 6, 'state': '11111', 'current_trial': 0, 'current_demo': 0}
        # ]

        # self.test_arbitrary_state = np.array([[ 1.35,  1.65,  0.17,  0.53,  0.01],
        #                                     [ 1.39,  0.47,  0.14,  1.53,  1.75],
        #                                     [-0.8,   0.6,  -0.72,  0.62, -0.89],
        #                                     [ 0.55, -0.95,  0.65,  0.98, -0.91],
        #                                     [ 1.,    1.,    1.,    1.,    1. ]])  # 没有选择起始点作为test key frames
        
        self.test_arbitrary_state = np.array([[ 1.35,  1.65,  0.17,  0.53,  0.01,   1.98,  1.83,  1.33,  1.12,  0.59],
                                            [ 1.39,  0.47,  0.14,  1.53,  1.75, 0.18,  0.53,  0.94,  1.05,  1.38],
                                            [-0.8,   0.6,  -0.72,  0.62, -0.89, -0.11, -0.49, -0.73, -0.66,  0.1],
                                            [ 0.55, -0.95,  0.65,  0.98, -0.91, 0.54,  0.67,  0.4,   0.29,  0.01],
                                            [ 1.,    1.,    1.,    1.,    1.,   1.,    1.,    1.,    1.,    1.    ]])
        

        self.test_arbitrary_state = np.array([[ 1.35,  1.65,  0.17,  0.53,  0.01],
                                            [ 1.39,  0.47,  0.14,  1.53,  1.75],
                                            [-0.8,   0.6,  -0.72,  0.62, -0.89],
                                            [ 0.55, -0.95,  0.65,  0.98, -0.91],
                                            [ 1.,    1.,    1.,    1.,    1.]])
        
        self.teaching_arbitrary_state = np.array([[ 1.75,  0.22,  1.77,  0.42,  1.08],
                                                    [ 0.53,  1.66,  0.02,  0.34,  1.13],
                                                    [ 0.95, -0.85, -0.86,  0.51,  0.89],
                                                    [-0.72, -0.22,  0.5,  -0.34,  0.88],
                                                    [ 1.,    1.,    1.,    1.,    1.  ]])
        

        intro_state = np.array([[ 0.72,   0.87,  1.40,  0.12,  1.33],
                                [ 1.34,  0.42,  0.26,   0.63,   0.73],
                                [ 0.54, -0.04,  0.29,  -0.24, -0.17],
                                [-0.20,  0.89,   -0.15, -0.02, -0.15],
                                [ 1.,1.,1.,1.,1.]])
        
        intro_state = np.array(    [[ 0.62,  1.39,  0.76,  0.36,  0.06],
                                    [ 0.13,  1.39,  0.91,  1.07,  1.79],
                                    [ 0.98, -0.97,  0.33, -0.47, -0.96  ],
                                    [ 0.52, -0.66,  -0.23,  0.18,  0.66],
                                    [ 1.,          1.,          1.,          1.,          1.        ]])


        intro_state2 = np.array(   [[ 0.30,  1.75,  0.60,  0.02,  1.13],
                                    [ 1.86,  0.57,  1.79,  0.73,  0.99 ],
                                    [ 0.23,  -0.05,  0.37,  0.52,  0.48],
                                    [ 0.85, -0.82, -0.31,   0.84, -0.42],
                                    [ 1.,          1.,          1.,          1.,          1.        ]])


        self.teaching_meaningful_state = np.array( [[ 2.,    1.92,  1.71,  1.33,  0.61],
                                                    [ 0.,    0.36,  0.66,  0.94,  1.27],
                                                    [ 0.,   -0.3,  -0.62, -0.73, -0.19],
                                                    [ 0.,    0.66,  0.62,  0.4,   0.13],
                                                    [ 1.,    1.,    1.,    1.,    1.]])
        

        self.generalization_state = np.array([  [ 0.08,  1.56,  0.19,  1.27,  0.03 ],
                                                [ 1.87,  0.85,  0.51,   0.38,  0.83],
                                                [-0.29,  -0.49, -0.22,  0.7,  0.18],
                                                [ 0.29,  -0.24, 0.20,  -0.13, -0.22],
                                                [ 1.,    1.,    1.,    1.,      1.]])
        
        self.generalization_state = np.array([  [ 0.72,   0.87,  1.39,  0.12,  1.33],
                                                [ 1.34,  0.42,  0.26,   0.63,   0.73],
                                                [ 0.14, -0.12,  0.98, -0.79, -0.58],
                                                [-0.68,  0.31, -0.49, -0.07, -0.51],
                                                [ 1.,    1.,    1.,    1.,      1.]])
        

        # # Formal experiment
        # generalization target group
        self.phases = [
            {'trials': 1, 'demos_per_trial': 5, 'teaching_state': False, 'current_trial': 1, 'current_demo': 1, 'given_phi': self.test_arbitrary_state, 'change_ordering': False},
            {'trials': 1, 'demos_per_trial': 5, 'teaching_state': False, 'current_trial': 1, 'current_demo': 1, 'given_phi': self.generalization_state, 'change_ordering': False},
            {'trials': 8, 'demos_per_trial': 5, 'teaching_state': True, 'current_trial': 1, 'current_demo': 1, 'given_phi': self.teaching_arbitrary_state, 'change_ordering': True},
            {'trials': 1, 'demos_per_trial': 5, 'teaching_state': False, 'current_trial': 1, 'current_demo': 1, 'given_phi': self.test_arbitrary_state, 'change_ordering': False},
            {'trials': 1, 'demos_per_trial': 5, 'teaching_state': False, 'current_trial': 1, 'current_demo': 1, 'given_phi': self.generalization_state, 'change_ordering': False}
        ]


        self.current_phase = 0  # Current stage

        self.force_teach_flag = force_teach
        self.state_teach_flag = state_teach
        self.plot_result_figure_flag = plot_result_figure_flag
        self.std_noise = std
        self.guidancetype = show_guidace_flag

        self.student_animation_show_flag = student_animation

        # data save initialization
        self.records = {
            'datas': [],
            'count': 0
        }
        
        self.ordering = np.array([[0, 1, 2, 3, 4],
                                [2, 0, 1, 4, 3],
                                [0, 2, 3, 1, 4],
                                [3, 0, 1, 2, 4],
                                [0, 4, 2, 1, 3],
                                [4, 1, 3, 0, 2],
                                [1, 4, 2, 0, 3],
                                [3, 1, 4, 0, 2]])
        # 8 trials
    
        self.teacher_value()
        # self.state_check_before_exp(self.phases[0]['given_phi'], self.phases[1]['given_phi'], self.phases[2]['given_phi'], self.phases[3]['given_phi'], self.phases[4]['given_phi'])

    def state_check_before_exp(self, *states):
        sns.set_style('darkgrid')
        num_states = len(states)  # 确定有多少个state输入
        fig, axs = plt.subplots(1, num_states, figsize=(6*num_states, 4), sharex=True)  # 动态调整子图数量和大小

        # 如果只有一个state，将axs转换为列表，以便下面的代码可以统一处理
        if num_states == 1:
            axs = [axs]

        for i, state in enumerate(states):
            # print(states)
            # print('determinant:', np.linalg.det(normalize(state, axis=0, norm="l2")))
            if i == 0 or i == 2 or i == 3:
                force = self.theta_1 @ state
            elif i == 1 or i == 4:
                force = self.theta_2 @ state
            print("optimal force of each state\n", force)
            force_min, force_max = np.amin(force), np.amax(force)

            # if np.min(force) < -0.5 or np.max(force) > 0.5:
            #     print(f"state {i+1}'s force out of range.")

            # 绘制位置
            axs[i].scatter(state[0,:], state[1,:], color='red', marker='o', label='fixed state')
            
            ratio = 0.1
            # 添加箭头
            for j in range(np.shape(state)[1]):
                axs[i].arrow(state[0,j], state[1,j], state[2,j] * ratio, state[3,j] * ratio, head_width=0.1, head_length=0.1, color='blue', overhang=0.6)
                axs[i].arrow(state[0,j], state[1,j], force[0,j] * ratio, force[1,j] * ratio, head_width=0.1, head_length=0.1, color='green', overhang=0.6)
                
            # 添加标签
            labels = ['1', '2', '3', '4', '5'][:np.shape(state)[1]]  # 确保labels不超过state的数量
            for label, xi, yi in zip(labels, state[0,:], state[1,:]):
                axs[i].annotate(label, (xi, yi), textcoords="offset points", xytext=(0,10), ha='center')

            axs[i].set_title(f'Phase {i+1}')
            # 画网格代码
            axs[i].xaxis.set_major_locator(plt.MultipleLocator(1.0))
            axs[i].yaxis.set_major_locator(plt.MultipleLocator(1.0))
            axs[i].xaxis.set_minor_locator(plt.MultipleLocator(0.2))
            axs[i].yaxis.set_minor_locator(plt.MultipleLocator(0.2))
            axs[i].grid(which='minor', color='gray', linestyle=':')
            axs[i].grid(which='major', color='gray', linestyle=':')

            axs[i].xaxis.set_ticks([])  # 隐藏x轴刻度
            axs[i].yaxis.set_ticks([])


            # 底部添加force信息
            plt.figtext((i+1)/(num_states+1), 0.01, f'phase {i+1} force: ({force_min:.3f}, {force_max:.3f})', ha='center', va='bottom', fontsize=12)

        plt.show()
        
    def teacher_value(self):
        teacher_robot = BasicRobotArm(link1_length=1, link2_length=1, 
                        link1_mass=1, link2_mass=1, 
                        joint1_angle=0.0, joint2_angle=0.0, 
                        joint1_velocity=0.0, joint2_velocity=0.0, 
                        joint1_torque=0.0, joint2_torque=0.0,
                        time_step=0.05, g=9.81)
        
        self.teacher1 = TeacherRobot(teacher_robot,0.2, 0.3, 0.8, 1.2)
        self.teacher1.teacher_demonstration(False, update_step=200, L_kd=self.theta_1)
        self.teacher1_trajectory = self.teacher1.trajectory
        self.teacher1.robot.reset()

        self.teacher2 = TeacherRobot(teacher_robot,0.2, 0.3, 0.8, 1.2)
        self.teacher2.teacher_demonstration(False, update_step=200, L_kd=self.theta_2)
        self.teacher2_trajectory = self.teacher2.trajectory

        print("self.teacher1_trajectory", type(self.teacher1_trajectory))
        print("self.teacher2_trajectory", np.shape(self.teacher2_trajectory))
        self.real_trajectory = self.teacher1_trajectory

        # plt.plot(np.array(self.teacher1_trajectory)[:, 0], np.array(self.teacher1_trajectory)[:, 1], 'r--')
        # plt.plot(np.array(self.teacher2_trajectory)[:, 0], np.array(self.teacher2_trajectory)[:, 1], 'r--')
        # plt.title("trajectory check")
        # plt.show()


    def reset_canvas_config(self):        
        self.simulator.reset_robot()
        # self.simulator.draw_arm(ani=False)
        # self.set_given_state()
        self.set_robot_state()

    def reset_teaching_config(self):
        # self.demo_cnt = 0

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

        
    def create_canvas(self):
        self.plot_frame = tk.Frame(self)
        # self.plot_frame.grid(row=1, column=1, padx=20, pady=5)
        self.plot_frame.grid(row=0, column=0, sticky="NESW")
        self.plot_frame.configure(background='white')

        self.simulator = RobotArmSimulator(self.plot_frame, self.robot)
        self.simulator.draw_arm(ani=False)

    def create_intro(self):
        self.intro_frame = tk.Frame(self)
        # self.intro_frame.grid(row=1, column=0, padx = 104, pady = 20)
        self.intro_frame.grid(row=1, column=0)
        self.intro_frame.configure(background='white')

        self.skill1_demo_btn = tk.Button(self.intro_frame, text="Skill 1 Movement", height=2, width=33, font=("Arial", 12), command=self.skill1_demo_callback)
        self.skill1_demo_btn.grid(row=0, column=0, sticky="nsew", pady = 20)
        self.skill1_demo_btn.config(state='disabled')

        self.skill2_demo_btn = tk.Button(self.intro_frame, text="Skill 2 Movement", height=2, width=33, font=("Arial", 12), command=self.skill2_demo_callback)
        self.skill2_demo_btn.grid(row=0, column=1, sticky="nsew", pady = 20) #pady = 5)
        self.skill2_demo_btn.config(state='disabled')
 
    def create_user_info_input(self):
        self.user_info_frame = tk.Frame(self)
        self.user_info_frame.grid(row=2, column=0) #, pady=140, sticky=tk.EW)
        self.user_info_frame.configure(background='white')

        user_entry_label = tk.Label(self.user_info_frame, text='Participant Number : ', background='white')
        user_entry_label.grid(row=0, column=0)

        self.user_entry = tk.Entry(self.user_info_frame)
        self.user_entry.grid(row=0, column=1, padx = 10)
        if self.administrator_mode == True:
            self.user_entry.insert(0, 0)
        self.user_entry.insert(0, 0)

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
        self.exp_start_frame.grid(row=3, column=0)
        self.exp_start_frame.configure(background='white')

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

        target_time = datetime.datetime.now() + datetime.timedelta(minutes=2) # 1 min counterdown clock gaishijian
        target_time = datetime.datetime.now() + datetime.timedelta(seconds=0) # 10 se   conds counterdown cl ock

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

            self.phase_init()
            self.create_progress_bar()
            self.create_button()
            self.reset_canvas_config()


            self.simulator.mouse_input(given_states=self.phases[self.current_phase]['given_phi'], 
                                       optimal_force = self.theta_1 @ self.phases[self.current_phase]['given_phi'])            
            
            self.skill_init()
            # self.phase_selection(phase_num=self.phase_cnt)
            self.btn1.config(state='normal')
            self.set_robot_state()
        else:
            messagebox.showerror('Please input a participant ID.')

    def create_progress_bar(self):
        progress_frame = tk.Frame(self)
        # progress_frame.grid(row=2, column=0, columnspan = 2)
        progress_frame.grid(row=2, column=0)
        progress_frame.configure(background='white')
        current_phase_data = self.phases[self.current_phase]
        
        # self.progress_bar.reset(num = current_phase_data['demos_per_trial'])
        
        self.progress_bar = ProgressBar(progress_frame, steps=current_phase_data['demos_per_trial'], start_step=0)
        # self.progress_bar = ProgressBar(progress_frame, start_step=0)
        self.progress_bar.pack(pady=20)

    def phase_init(self):
        current_phase_data = self.phases[self.current_phase]
        if current_phase_data['teaching_state'] == True:
            # self.update_page(True)
            pass
    
    def create_button(self):
        button_frame = tk.Frame(self)
        # button_frame.grid(row=1, column=0, columnspan=2) #, pady=5, sticky=tk.EW)
        button_frame.grid(row=1, column=0)
        button_frame.configure(background='white')

        self.btn1 = tk.Button(button_frame, text="OK", height=2, width=11, font=("Arial", 12), command=self.on_btn1_clicked)
        self.btn1.grid(row=0, column=0, padx = 10)
        self.btn1.config(state='disabled')

        self.btn2 = tk.Button(button_frame, text="Next", height=2, width=11, font=("Arial", 12), command=self.on_btn2_clicked)
        self.btn2.grid(row=0, column=1, padx = 10)
        self.btn2.config(state='disabled')

    def on_btn1_clicked(self):
        # self.simulator.arrow_drawer.confirm_arrow_and_prepare_next()
        done = self.simulator.arrow_drawer.confirm_arrow()
        if done:
            self.btn2.config(state="normal")
            current_phase_data = self.phases[self.current_phase]
            
            if current_phase_data['current_demo'] <= current_phase_data['demos_per_trial']+1:
                if self.administrator_mode:
                    self.btn1.config(state="normal")
                    self.btn2.config(state="normal")
                    
                else:
                    self.btn1.config(state="disabled")
                    self.btn2.config(state="normal")
                    

                x = self.simulator.robot.get_end_effector_position()[0]
                y = self.simulator.robot.get_end_effector_position()[1]
            
                self.phi_recording = self.set_states[:, 0: current_phase_data['current_demo']]
                phi_single = self.set_states[:, current_phase_data['current_demo']-1]
                
                # if self.force_teach_flag:
                #     self.force_recording.append([self.force_slider_x.get()/self.slider_scale, self.force_slider_y.get()/self.slider_scale])

                # # force diff as guidacne 真实值-用户值
                # force_single = np.array(self.force_recording[-1])
                # real_force_single = self.real_theta @ phi_single
                # force_guidance = real_force_single - force_single

                
                if current_phase_data['teaching_state'] == True and self.guidancetype:
                    pass
                elif current_phase_data['teaching_state'] == False:
                    pass

    def on_btn2_clicked(self):
        self.simulator.arrow_drawer.prepare_next()
    
        current_phase_data = self.phases[self.current_phase]
        
        if current_phase_data['teaching_state'] == True and self.guidancetype:
            pass

        if current_phase_data['current_demo'] <= current_phase_data['demos_per_trial']:
            current_phase_data['current_demo'] += 1  # demo 加1
            self.btn1.config(state="normal")
            self.btn2.config(state='disabled')

            
            
            
            self.progress_bar.advance_progress()

            if current_phase_data['current_demo'] <= current_phase_data['demos_per_trial']:
                self.set_robot_state()

            if current_phase_data['current_demo'] == current_phase_data['demos_per_trial']+1: # demo达到每个trail最大值，重置demo等于1
                current_phase_data['current_demo'] = 1  # 重置演示
    
                self.btn1.config(state="disabled")
                messagebox.showinfo("Information", "This is what we learned.")
                self.teaching()
                

                self.btn1.config(state="normal")

                if current_phase_data['current_trial'] == current_phase_data['trials']:     #trial达到每个phase的最大值，开始判断需不需要变phase
                    if self.current_phase < len(self.phases)-1:                             # phase没达到最大值，phase 加1，到达下一个phase，达到下一个phase的时候可能需要改变技能？
                        self.current_phase += 1
                        if self.current_phase  == 1 or self.current_phase  == 4:
                            # self.simulator.starting_point.set_data([], [])
                            self.simulator.target_point.set_data([], [])
                            self.simulator.target_line.set_data([0.8, 0.8], [-0.1, 2.1])
                            self.real_theta = self.theta_2
                            self.real_trajectory = self.teacher2_trajectory
                        else:
                            self.simulator.target_point.set_data(0.8, 1.2)
                            self.simulator.target_line.set_data([], [])
                            self.real_theta = self.theta_1
                            self.real_trajectory = self.teacher1_trajectory


                        # current_phase_data = self.phases[self.current_phase]
                        # self.progress_bar.reset(num = current_phase_data['demos_per_trial'])

                        if current_phase_data['teaching_state'] == True:
                            # self.update_page(True)
                            pass
                    else:                                                                   # phase到最大值，全部实验结束
                        # 也许这里可以处理完成所有阶段的逻辑
                        print("All phases completed.")
                        messagebox.showinfo("Information", "Experiment Finished.")
                        self.quit()
                else:
                    current_phase_data['current_trial'] += 1

                current_phase_data = self.phases[self.current_phase]
                self.set_states = current_phase_data['given_phi']
                self.progress_bar.reset(num = current_phase_data['demos_per_trial'])
                
                if current_phase_data['change_ordering']:
                    # print("change ordering ")
                    order = self.ordering[current_phase_data['current_trial']-1]
                    self.set_states = self.set_states[:, order]

                try:
                    self.simulator.arrow_drawer.clear_all_elements()
                    self.simulator.arrow_drawer.given_states = self.set_states
                    self.simulator.arrow_drawer.optimal_force = (self.theta_1 @ self.set_states).T
                    self.simulator.arrow_drawer.reinit()
                    self.simulator.arrow_drawer.show_optiaml_force = current_phase_data['teaching_state']
                except AttributeError:
                    pass

                self.reset_canvas_config()
                return
             
    def skill_init(self):
        self.reset_teaching_config()
    
        self.simulator.starting_point.set_data([], []) # dont show starting point, 因为这是从任意一个点。
        self.simulator.target_point.set_data(0.8, 1.2)

        self.set_theta(1)

        self.teacher_trajectory = self.teacher1.trajectory
    
        self.simulator.robot.set_skill_theta(self.real_theta)
        self.set_robot_state()
        self.simulator.draw_arm(ani=False, show_arm=False)

    def set_theta(self, skill_num):
        if skill_num == 1:
            self.starting_point = [0.2, 0.3]
            self.target_point = [0.8, 1.2]
            self.step = 200
            self.real_theta = self.theta_1
            self.simulator.reset_robot()
        elif skill_num == 2:
            self.starting_point = [0.2, 0.3]
            self.target_point = [0.8, 0.641]
            self.step = 220
            self.real_theta = self.theta_2
            self.simulator.reset_robot()
        else:
            raise ValueError('Invalid Skill Number.')
    
    def set_robot_state(self):
        current_phase_data = self.phases[self.current_phase]
        selected_column = current_phase_data['current_demo']-1
        self.set_states = current_phase_data['given_phi']

        if current_phase_data['change_ordering']:
            # print("change ordering ")
            order = self.ordering[current_phase_data['current_trial']-1]
            self.set_states = self.set_states[:, order]

        if self.state_teach_flag == False:
            # if self.demo_cnt <= self.max_demo_num:
            if current_phase_data['current_demo'] <= current_phase_data['demos_per_trial']:

                x_position = self.set_states[0, selected_column]
                y_position = self.set_states[1, selected_column]
                x_velocity = self.set_states[2, selected_column]
                y_velocity = self.set_states[3, selected_column]
            
                joint_angles = self.simulator.robot.inverse_kinematics(x_position, y_position)
                self.simulator.robot.joint1_angle, self.simulator.robot.joint2_angle = joint_angles[0], joint_angles[1]
                self.simulator.draw_arm(x_velocity, y_velocity, 0,0)
        elif self.state_teach_flag == True:
            
            self.simulator.draw_arm(self.position_slider_x.get()/self.slider_scale, self.position_slider_y.get()/self.slider_scale, 0, 0)

    def teaching(self):
        current_phase_data = self.phases[self.current_phase]
        
        if len(self.phi_recording) == 0:
            messagebox.showwarning('Warning', 'No demonstration has recorded.')
            return
        # elif len(self.phi_recording) != current_phase_data['demos_per_trial']:
        #     messagebox.showwarning('Warning', 'Incomplete demonstration.')
            
        phi = np.array(self.phi_recording)
        if self.force_teach_flag == False: # force from calculated with noise
            force_real = self.real_theta @ phi
            force_noi_diff = self.force_std_noise * np.random.randn(2, 5)
            force = force_real + force_noi_diff
        elif self.force_teach_flag == True:
            force_teached = np.array(self.force_recording).T # np.shape (5,2)
            force = force_teached

        xf, yf = self.simulator.on_button_release(None)
        force = np.array([xf, yf])
        print("user choice force: \n", force)

        self.learned_theta = self.simulator.learn(phi, force)
        print(f"Phase: self.real_theta = {self.real_theta[0]}, {self.real_theta[1]}")
        print(f"Phase: self.learned_theta = {self.learned_theta[0]}, {self.learned_theta[1]}")


        self.simulator.robot.set_skill_theta(self.learned_theta)
        # show student's animation
        self.btn2.config(state='disabled')
        
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
        # rmse = np.sqrt(np.mean((np.array(self.teacher_trajectory) - np.array(self.student_trajectory))**2))
        rmse = np.sqrt(np.mean((np.array(self.real_trajectory) - np.array(self.student_trajectory))**2))

        # print("self.real_theta,", self.real_theta)
        el2, nmse_x, nmse_y = self.error_calculator(self.real_theta, self.learned_theta)
        print("\nerror:")
        print(f"rmse: {rmse}, nmse_x: {nmse_x}, nmse_y: {nmse_y}, el2: {el2}")
        
        # calculate score, but only show on teaching phase
        real_force = self.real_theta @ phi
        force_guidance = real_force - force
        
        # 为了探究不同的lamba的影响，记录数据
        learned_theta_lambda6 = self.simulator.learn_lambda6(phi, force)
        el2_lambda6, nmse_x_lambda6, nmse_y_lambda6 = self.error_calculator(self.real_theta, learned_theta_lambda6)
        rmse_6 = 0.0

        # store data tocsv file
        self.store_data_tojson(phi, real_force, force, self.real_theta, self.learned_theta, force_guidance, el2, rmse, nmse_x, nmse_y, 
                               learned_theta_lambda6, el2_lambda6, rmse_6, nmse_x_lambda6, nmse_y_lambda6)

        # self.plot_user_result(phi, force)

        # reset robot
        self.simulator.reset_robot()
        self.reset_teaching_config()
        self.btn1.config(state='normal')

    
    def error_calculator(self, real_theta, learned_theta):
        el2 = self.el2_cal(real_theta, learned_theta)
        nmse_x, nmse_y = self.nmse_cal(real_theta, learned_theta)
        return el2, nmse_x, nmse_y
    
    def nmse_cal(self, real_theta, learned_thtea):
        test_phi = np.array([[1.54547433, 1.82535471, 0.20575465, 0.40244344, 0.09569714, 1.04273663, 1.72299245, 0.95102198, 0.68314063, 0.61546897],
                            [0.08959529, 0.01713778, 1.64400987, 0.53837856, 0.17259851, 1.47867964, 0.31134094, 0.59224916, 0.6820265, 1.03555592],
                            [-0.12833334, -0.10372383, -0.00663499, 0.1608878, -0.10024024, -0.26926053, -0.40315198, 0.23943124, -0.1348775, 0.05278681],
                            [0.15063608, 0.39135257, 0.16674541, -0.07797693, 0.38231418, 0.25929959, -0.25023862, 0.44598129, -0.46304699, -0.47250088],
                            [1,1,1,1,1,1,1,1,1,1]])
        n = test_phi.shape[1]
        
        force_desired = np.dot(real_theta, test_phi)
        force_learned = np.dot(learned_thtea, test_phi)

        nmse_x = np.mean((force_learned[0, :]-force_desired[0, :])**2) / np.var(force_desired[0, :], ddof=0)
        nmse_y = np.mean((force_learned[1, :]-force_desired[1, :])**2) / np.var(force_desired[1, :], ddof=0)
        
        print("nmse:", nmse_x, nmse_y)
        return nmse_x, nmse_y

    def el2_cal(self, real_theta, learned_thtea):
        el2 = np.linalg.norm(real_theta - learned_thtea)
        return el2


    def store_data_tojson(self, phi, real_force, user_force, real_theta, learned_theta, force_guidance, el2, rmse, nmse_x, nmse_y, learned_theta_6, el2_6, rmse_6, nmse_x_6, nmse_y_6):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        target_dir = os.path.join(current_dir, f'formal_results')
        filename = os.path.join(target_dir, f"formal_exp_{self.user_number}.json")

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        if os.path.exists(filename):
            with open(filename, 'r') as f:
                self.records = json.load(f)

        self.records['datas'].append({
            'phi': phi,
            'real_force': real_force,
            'user_force': user_force,
            'real_theta': real_theta,
            'learned_theta': learned_theta,
            'force_score': force_guidance,
            'el2': el2,
            'rmse': rmse,
            'nmse_x': nmse_x,
            'nmse_y': nmse_y,

            'learned_theta_lambda6': learned_theta_6, 
            'el2_lambda6': el2_6,
            'rmse_lambda6': rmse_6,
            'nmse1_lambda6': nmse_x_6, 
            'nmse2_lambda6': nmse_y_6, 
        })
        self.records['count'] += 1
        
        with open(filename, 'w') as f:
            json.dump(self.records, f, cls=NumpyEncoder)
        
    def skill1_demo_callback(self):
        self.skill1_demo_btn.config(state='disabled')
        self.skill2_demo_btn.config(state='disabled')
        self.set_theta(1)

        self.simulator.target_point.set_data([0.8], [1.2])

        starting_pts = [[0.2, 0.3], [0.3, 1.3], [0.7, 0.1], [1.6, 1.3], [0.1, 0.5]]
        for num in range(len(starting_pts)):
            self.simulator.init_x0, self.simulator.init_y0 = starting_pts[num]
            self.simulator.starting_point.set_data([self.simulator.init_x0], [self.simulator.init_y0])
            self.simulator.reset_robot()
            self.simulator.robot.set_skill_theta(self.real_theta)

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
        self.set_theta(2)

        # self.simulator.starting_point.set_data(0.2, 0.3)
        self.simulator.target_line.set_data([0.8, 0.8], [-0.1, 2.1])

        starting_pts = [[0.2, 0.3], [0.3, 1.5], [1.6, 1.3], [1.4, 0.5],[0.6, 0.8]]
        for num in range(len(starting_pts)):
            print("numnumnumnum", num)
            self.simulator.init_x0, self.simulator.init_y0 = starting_pts[num]
            self.simulator.starting_point.set_data([self.simulator.init_x0], [self.simulator.init_y0])
            self.simulator.reset_robot()
            self.simulator.robot.set_skill_theta(self.real_theta)

            x, y = self.simulator.robot.get_end_effector_position()
            # while (x-self.target_point[0])**2 > 0.0005:
            while (x-0.8)**2 > 0.0000005:
                self.simulator.robot.trajectory.append(list(self.simulator.robot.get_end_effector_position()))
                self.simulator.robot.velocity.append(list(self.simulator.robot.get_end_effector_velocity()))

                self.simulator.draw_arm()
                self.simulator.robot.update_rk4()
                x, y = self.simulator.robot.get_end_effector_position()
        
        self.simulator.init_x0, self.simulator.init_y0 = 0.2, 0.3
        self.simulator.starting_point.set_data([], [])
        self.simulator.target_point.set_data([], [])
        self.simulator.target_line.set_data([], [])

        self.simulator.reset_robot()

        self.skill1_demo_btn.config(state='normal')
        self.skill2_demo_btn.config(state='normal')

    def skill2_demo_callback_singlepoint(self):
        self.skill1_demo_btn.config(state='disabled')
        self.skill2_demo_btn.config(state='disabled')
        self.simulator.init_x0, self.simulator.init_y0 = 0.2, 0.3
        self.simulator.reset_robot()
        self.set_theta(2)
        self.simulator.robot.set_skill_theta(self.real_theta)

        self.simulator.starting_point.set_data(0.2, 0.3)
        # self.simulator.target_point.set_data(0.8, 0.641)
        self.simulator.target_line.set_data([0.8, 0.8], [-0.1, 2.1])

        for i in range(self.step):
            self.simulator.robot.trajectory.append(list(self.simulator.robot.get_end_effector_position()))
            self.simulator.robot.velocity.append(list(self.simulator.robot.get_end_effector_velocity()))

            self.simulator.draw_arm()
            self.simulator.robot.update_rk4()

        self.simulator.starting_point.set_data([], [])
        self.simulator.target_point.set_data([], [])
        self.skill1_demo_btn.config(state='normal')
        self.skill2_demo_btn.config(state='normal')

        self.simulator.init_x0, self.simulator.init_y0 = 0.2, 0.3
        # self.simulator.init_x0, self.simulator.init_y0 = 0.2111, 0.3111
        self.simulator.reset_robot()

    def plot_user_result(self, phi, force):
        # path = f'User{self.user_number}-Phase{self.phase_cnt}-Trial{self.trail_cnt}-Choices'
        path = f'1111111111111111111111111'
        self.plot_function(phi, force, path)

        path2 = f'222222222222222222222222'
        self.plot_dynamic_motion(path2)
        
    def plot_function(self, state, force, path):
        state_matrix = np.array(state)

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
            plt.arrow(state_matrix[0,i], state_matrix[1,i], state_matrix[2,i], state_matrix[3,i], head_width=0.1, head_length=0.1, color='blue', overhang=0.6)
            plt.arrow(state_matrix[0,i], state_matrix[1,i], force[0,i], force[1,i], head_width=0.1, head_length=0.1, color='red', overhang=0.6)
            
        labels = ['1', '2', '3', '4', '5']
        for label, xi, yi in zip(labels, state_matrix[0,:], state_matrix[1,:]):
            plt.annotate(label, (xi, yi), textcoords="offset points", xytext=(0,10), ha='center')

        plt.arrow(-0.3, 0, 2.5, 0, head_width=0.1, head_length=0.1, color='black', overhang=0.2)  # 参数依次是起始点坐标、箭头的水平长度、箭头头部的宽度和长度，颜色等
        plt.arrow(0, -0.3, 0, 2.5, head_width=0.1, head_length=0.1, color='black', overhang=0.2)
        
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
        fig.suptitle(f'Dynamics Motion \n User')

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

    # app = RobotArmApp(robot=robot, demo_num=5, trial_num=1, pilot_num=3, force_teach=True, state_teach=True, plot_result_figure_flag=False, std=0.0, 
    #                   show_guidace_flag = True, student_animation = True, administrator_mode = True, visual_guidacne=True)
    
    app = RobotArmApp(robot=robot, demo_num=5, trial_num=1, pilot_num=3, force_teach=True, state_teach=False, plot_result_figure_flag=False, std=0.0, 
                      show_guidace_flag = True, student_animation = True, administrator_mode = False, visual_guidacne=False)


    app.mainloop()
