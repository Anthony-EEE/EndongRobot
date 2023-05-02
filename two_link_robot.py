import numpy as np

import matplotlib.pyplot as plt

from utils import BasicRobotArm

from scipy.fft import fft, ifft, fftfreq

class RobotArm(BasicRobotArm):
    def __init__(self, 
                 link1_length, link2_length, link1_mass, link2_mass,
                 joint1_angle=0.0, joint2_angle=0.0, joint1_velocity=0.0, joint2_velocity=0.0, joint1_torque=0.0, joint2_torque=0.0, 
                 time_step=0.01, ifdemo=True, g=9.81):
        """
        Initializes a 2-link planar robot arm object with given physical parameters and initial joint angles,
        velocities, torques, time step, gravity and demonstration flag.

        Inputs:
        - link1_length: float, length of link 1 (m)
        - link2_length: float, length of link 2 (m)
        - link1_mass: float, mass of link 1 (kg)
        - link2_mass: float, mass of link 2 (kg)
        - joint1_angle: float, initial angle of joint 1 (rad)
        - joint2_angle: float, initial angle of joint 2 (rad)
        - joint1_velocity: float, initial velocity of joint 1 (rad/s)
        - joint2_velocity: float, initial velocity of joint 2 (rad/s)
        - joint1_torque: float, initial torque of joint 1 (N*m)
        - joint2_torque: float, initial torque of joint 2 (N*m)
        - time_step: float, time step for simulation (s)
        - ifdemo: bool, whether or not to record demonstrations during PD control
        - g: float, gravitational acceleration (m/s^2)
        
        Outputs:
        None
        """
        super().__init__(link1_length, link2_length, link1_mass, link2_mass, 
                 joint1_angle, joint2_angle, joint1_velocity, joint2_velocity, joint1_torque, joint2_torque,
                 time_step, g)
        
        # End-effector trajectory
        self.trajectory = []

        # Demonstrations
        self.ifdemo = ifdemo
        self.demonstrations = []

    def pd_controller(self, desired_position, desired_velocity, stiffness, damping):
        """
        Computes the joint torques required to achieve the desired end-effector position and velocity
        using a PD controller:

        joint_torques = J.T * (Kp * delta_position + Kd * delta_velocity)

        Inputs:
        - desired_position: 2x1 array representing the desired end-effector position [x, y]
        - desired_velocity: 2x1 array representing the desired end-effector velocity [v_x, v_y]
        - stiffness: 2x2 array representing the proportional gain matrix
        - damping: 2x2 array representing the derivative gain matrix

        Outputs:
        - joint_torques: 2x1 array representing the required joint torques [tau1, tau2]
        - delta_position: 2x1 array representing the error in end-effector position
        - delta_velocity: 2x1 array representing the error in end-effector velocity
        """
        # Calculate the current position and velocity of the end-effector
        current_position = self.forward_kinematics()
        current_velocity = np.dot(self.jacobian(), [self.joint1_velocity, self.joint2_velocity])

        # Calculate the errors in position and velocity
        delta_position = np.array(desired_position) - np.array(current_position)
        delta_velocity = np.array(desired_velocity) - np.array(current_velocity)

        # Calculate the joint torques required to produce the desired acceleration
        delta_phi = np.concatenate((delta_position, delta_velocity), axis=0)
        joint_torques = np.dot(np.concatenate((stiffness, damping), axis=1), delta_phi) #np.dot(self.jacobian().T, np.dot(np.concatenate((stiffness, damping), axis=1), delta_phi))

        # Update the joint torques of the robot object and return them
        self.joint1_torque = joint_torques[0]
        self.joint2_torque = joint_torques[1]

        # Update the joint angles of the robot
        self.update_joint_states()
        return joint_torques, delta_position, delta_velocity

    def circle(self, inital_q1q2 = [np.pi/3, np.pi/3], inital_q1dq2d = [0, 0], plot=True):
        self.set_joint_angles(inital_q1q2[0], inital_q1q2[1])
        self.set_joint_velocity(inital_q1dq2d[0], inital_q1dq2d[1])
        q1all = []
        q2all = []
        q1dall = []
        q2dall = []
        k = 0

        while k <= 10000:
        #while abs(self.joint2_angle - inital_q1q2[1]) < np.pi/5:
            k=k+1
            m1, m2 = self.link1_mass, self.link2_mass
            l1, l2 = self.link1_length, self.link2_length
            q1, q2 = self.joint1_angle, self.joint2_angle
            q1_dot, q2_dot = self.joint1_velocity, self.joint2_velocity

            # L = np.array([[(m1+m2)*self.g*l1, 0, m2*l1*l2, 0], [0, m2*self.g*l2, 0, -m2*l1*l2]])
            # L = np.array([[(m1+m2)*l1*self.g, m2*self.g*l2, 0, -m2*l1*l2], [0, m2*l2*self.g, m2*l1*l2, 0]])
            L = np.array([[-1, 0, 0, 0], [0, -1, 0, 0]])
            self.circle_L = L

            #joints = np.add(np.array(np.dot(L, np.array([q1, q2, q1_dot, q2_dot]).T)), np.array([-np.sin(q2)*(2*q1_dot*q2_dot + q2_dot*q2_dot), q1_dot*q1_dot*np.sin(q2)]))
            #joints = np.array(np.dot(L, np.array([q1, q2, q1_dot, q2_dot]).T)) + np.array([[-np.sin(q2)*(2*q1_dot*q2_dot + q2_dot*q2_dot)], [q1_dot*q1_dot*np.sin(q2)]])
            #joints = np.array(joints)+ np.array([[-np.sin(q2)*(2*q1_dot*q2_dot + q2_dot*q2_dot)], [q1_dot*q1_dot*np.sin(q2)]])            
            joints = np.dot(L, np.array([q1, q2, q1_dot, q2_dot]).T)
            #print(joints)
            #print(type(joints))

            self.joint1_torque = joints[0]
            self.joint2_torque = joints[1]

            if self.ifdemo:
                self.demonstrations.append([q1, q2, q1_dot, q2_dot, self.joint1_torque, self.joint2_torque])

            # Add the current end-effector position to the trajectory
            self.trajectory.append(self.get_end_effector_position())

            if plot:
                # Plot the current state of the robot arm and the trajectory
                plt.clf()
                plt.xlim(self.x_min, self.x_max)
                plt.ylim(self.y_min, self.y_max)
                plt.plot([0, self.link1_length * np.cos(self.joint1_angle), self.link1_length * np.cos(self.joint1_angle) + self.link2_length * np.cos(self.joint1_angle + self.joint2_angle)], [0, self.link1_length * np.sin(self.joint1_angle), self.link1_length * np.sin(self.joint1_angle) + self.link2_length * np.sin(self.joint1_angle + self.joint2_angle)], '-o')
                plt.plot([pos[0] for pos in self.trajectory], [pos[1] for pos in self.trajectory], '-r')
                plt.grid()
                plt.draw()
                plt.title('Teacher')
                plt.pause(0.001)

            q1all.append(q1)
            q2all.append(q2)
            q1dall.append(q1_dot)
            q2dall.append(q2_dot)

            self.update_joint_states()

        q1all = np.array(q1all)
        q2all = np.array(q2all)
        q1dall = np.array(q1dall)
        q2dall = np.array(q2dall)
        
        #q1fft = fft(q1all)
        #print(q1fft)
        
        # plot fft
        #N = 600
        #T = 1.0 / 800.0
        #yf = fft(q2dall)
        #xf = fftfreq(N, T)[:N//2]

        #plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
        #plt.grid()
        #plt.show()

        # plot q, qdot
        t = list(range(len(q1all)))
        fig, axs = plt.subplots(4)
        #fig.suptitle('L = [-1, 0, -1, 0], [0, -1, 0, -1]')
        fig.suptitle('{}'.format(L))
        #fig.suptitle('When L = {}, {}'.format(L[0], L[1]))
        
        axs[0].plot(t, q1all)
        axs[0].set(xlabel='t', ylabel='q1') 
        axs[1].plot(t, q2all)
        axs[1].set(xlabel='t', ylabel='q2')
        axs[2].plot(t, q1dall)
        axs[2].set(xlabel='t', ylabel='q1_dot')
        axs[3].plot(t, q2dall)
        axs[3].set(xlabel='t', ylabel='q2_dot')
        
        plt.show()



        

    def move_to(self, desired_position, desired_velocity, stiffness=np.diag([5.0, 5.0]), damping=np.diag([1.0, 1.0]), threshold=[0.01, 0.1], max_iterations=1000, plot=False):
        """
        Moves the end-effector of the robot to a desired position using PD control.
        
        Inputs:
        - desired_position: a list or array of length 2 containing the desired x and y position of the end-effector
        - desired_velocity: a list or array of length 2 containing the desired x and y velocity of the end-effector
        - stiffness: a 2x2 numpy array containing the gains for the PD controller for position control. The default is [5.0, 5.0] for both joints.
        - damping: a 2x2 numpy array containing the gains for the PD controller for velocity control. The default is [1.0, 1.0] for both joints.
        - threshold: a list or array of length 2 containing the maximum allowed error in position and velocity for the end-effector.
        - max_iterations: maximum number of iterations for the PD control loop.
        - plot: a boolean indicating whether or not to plot the trajectory of the end-effector during the motion.
        
        Outputs:
        None
        """
        x, y = desired_position[0], desired_position[1]

        # Initialize variables
        end_effector_position = np.array([x, y])
        end_effector_velocity = np.array([desired_velocity[0], desired_velocity[1]])
        iteration = 0

        # PD control loop
        while True:
            # Calculate current end-effector position and velocity
            current_position = self.forward_kinematics()
            current_velocity = np.dot(self.jacobian(), [self.joint1_velocity, self.joint2_velocity])

            # Check if goal is reached: position delta < threshold and velocity delta < threshold or loop get the max.
            if np.linalg.norm(end_effector_position - np.array(current_position)) < threshold[0] and np.linalg.norm(end_effector_velocity - np.array(current_velocity)) < threshold[1] or iteration >= max_iterations:
                break

            # Compute desired position and velocity using PD controller
            desired_position = np.array([x, y])
            desired_velocity = desired_velocity

            # Calculate the joint torques required to produce the desired acceleration
            joint_torques, delta_position, delta_velocity = self.pd_controller(desired_position, desired_velocity, stiffness, damping)

            # Record the second loop because of its better trajectory
            if self.ifdemo:
                self.demonstrations.append([delta_position[0], delta_position[1], delta_velocity[0], delta_velocity[1], joint_torques[0], joint_torques[1]])

            # Add the current end-effector position to the trajectory
            self.trajectory.append(self.get_end_effector_position())

            if plot:
                # Plot the current state of the robot arm and the trajectory
                plt.clf()
                plt.plot(desired_position[0], desired_position[1], '*', c='green')
                plt.xlim(self.x_min, self.x_max)
                plt.ylim(self.y_min, self.y_max)
                plt.plot([0, self.link1_length * np.cos(self.joint1_angle), self.link1_length * np.cos(self.joint1_angle) + self.link2_length * np.cos(self.joint1_angle + self.joint2_angle)], [0, self.link1_length * np.sin(self.joint1_angle), self.link1_length * np.sin(self.joint1_angle) + self.link2_length * np.sin(self.joint1_angle + self.joint2_angle)], '-o')
                plt.plot([pos[0] for pos in self.trajectory], [pos[1] for pos in self.trajectory], '-r')
                plt.grid()
                plt.draw()
                plt.title('Teacher')
                plt.pause(0.001)
            
            # Update iteration count
            iteration += 1
    
    def move_to_circle(self, center_x, center_y, radius, num_points=100, stiffness=np.diag([5.0, 5.0]), damping=np.diag([1.0, 1.0]), threshold=[0.01, 0.5], max_iterations=1000):
            """
            Moves the robot arm to points along a circle centered at (center_x, center_y) with the given radius.

            Inputs:
            - center_x: x-coordinate of the center of the circle
            - center_y: y-coordinate of the center of the circle
            - radius: radius of the circle
            - num_points: number of points to move the robot arm to along the circle (default: 100)
            - stiffness: 2x2 numpy array of stiffness gains for the PD controller (default: diagonal matrix with 5.0 on diagonal)
            - damping: 2x2 numpy array of damping gains for the PD controller (default: diagonal matrix with 1.0 on diagonal)
            - threshold: list of error thresholds to check if goal is reached in position and velocity (default: [0.01, 0.5])
            - max_iterations: maximum number of iterations before giving up on reaching the goal (default: 1000)
            
            Outputs:
            None
            """
            # Generate sequence of x and y values around the circle
            angles = np.linspace(0, 2*np.pi, num_points)
            x_vals = center_x + radius * np.cos(angles)
            y_vals = center_y + radius * np.sin(angles)

            # Initialize loop count
            loop = 0

            while loop < 3:
                # Call move_to() for each x, y value in the circle sequence
                for i in range(num_points):
                    # Check if end-effector position is within robot's workspace
                    if x_vals[i] < self.x_min or x_vals[i] > self.x_max or y_vals[i] < self.y_min or y_vals[i] > self.y_max:
                        raise ValueError("End-effector position outside of robot's workspace. Skipping...")
                    
                    # Move robot arm to desired end-effector position
                    self.move_to([x_vals[i], y_vals[i]], desired_velocity=[0, 0], stiffness=stiffness, damping=damping, threshold=threshold, max_iterations=max_iterations, plot=True)

                loop += 1
