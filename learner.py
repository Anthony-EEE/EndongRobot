import numpy as np
import matplotlib.pyplot as plt
from utils import BasicRobotArm

class Learner(BasicRobotArm):
    def __init__(self, S=4, lamb=1e-6, link1_length=0.5, link2_length=1.0, link1_mass=1.0, link2_mass=1.0,
                 joint1_angle=0.0, joint2_angle=0.0, joint1_velocity=0.0, joint2_velocity=0.0, joint1_torque=0.0, joint2_torque=0.0, 
                 time_step=0.01, g=9.81):
        """
        Initialize a Learner object.

        Args:
            S (int): The minimum number of training items or dimension of input features. Default is 4.
            lamb (float): Regularization coefficient. Default is 1e-6.
            link1_length (float): The length of the first link of the robot arm. Default is 0.5.
            link2_length (float): The length of the second link of the robot arm. Default is 1.0.
            link1_mass (float): The mass of the first link of the robot arm. Default is 1.0.
            link2_mass (float): The mass of the second link of the robot arm. Default is 1.0.
            joint1_angle (float): The initial angle of the first joint of the robot arm. Default is 0.0.
            joint2_angle (float): The initial angle of the second joint of the robot arm. Default is 0.0.
            joint1_velocity (float): The initial velocity of the first joint of the robot arm. Default is 0.0.
            joint2_velocity (float): The initial velocity of the second joint of the robot arm. Default is 0.0.
            joint1_torque (float): The initial torque of the first joint of the robot arm. Default is 0.0.
            joint2_torque (float): The initial torque of the second joint of the robot arm. Default is 0.0.
            time_step (float): The time step used in the simulation. Default is 0.01.
            g (float): The gravitational acceleration. Default is 9.81.
        """
        super().__init__(link1_length, link2_length, link1_mass, link2_mass, 
                 joint1_angle, joint2_angle, joint1_velocity, joint2_velocity, joint1_torque, joint2_torque,
                 time_step, g)
        
        # Initial states of the robot
        self.init_joint1_angle = joint1_angle
        self.init_joint2_angle = joint2_angle
        self.init_joint1_velocity = joint1_velocity
        self.init_joint2_velocity = joint2_velocity
        self.init_joint1_torque = joint1_torque
        self.init_joint2_torque = joint2_torque

        # End-effector trajectory
        self.trajectory = []

        # Target learnt parameters
        self.learnt_parameters = []
        self.learnt_parameter = None

        # regularization coeffcient
        self.lamb = lamb

        # Minimum number of training items or dimension of input features
        self.S = S
        
        # Identity matrix
        self.I = np.identity(S)

        # det(phi)
        self.det_phi = []

        # Errors
        self.rmse = []

        
    def reset_robot(self):
        """
        Reset the state of the robot to the inital states.
        """
        self.joint1_angle = self.init_joint1_angle
        self.joint2_angle = self.init_joint2_angle
        self.joint1_velocity = self.init_joint1_velocity
        self.joint2_velocity = self.init_joint2_velocity
        self.joint1_torque = self.init_joint1_torque
        self.joint2_torque = self.init_joint2_torque

    def learning_process(self, train_data):
        """
        Perform learning process using linear regression.

        Args:
            train_data (list): List of demonstrations, where each demonstration is a list of length S+2, where first S elements
            are features and last two elements are labels.

        Returns:
            numpy.ndarray: Learnt parameters for each iteration.
        """

        # Convert list of demonstrations to numpy array, shape: NxS
        train_data = np.array(train_data)

        # First S elements are features SxN i.e. 4xN
        self.phi = train_data[:, 0:self.S].T
        for i in range(self.phi.shape[1]):
            self.phi[:, i] = self.calcu_phi(self.phi[:, i][0], self.phi[:, i][1], self.phi[:, i][2], self.phi[:, i][3])

        # Last two elements are labels Nx2
        self.u = train_data[:, self.S:]

        # det(norma_phi)
        self.norm_phi = np.copy(self.phi)
        for i in range(self.phi.shape[1]):
            self.norm_phi[:, i] = self.phi[:, i] / np.linalg.norm(self.phi[:, i])
        self.det_phi.append(abs(np.linalg.det(self.norm_phi)))

        # Learning using linear regression
        self.learnt_parameter = np.linalg.inv(self.phi @np.transpose(self.phi) + self.lamb*self.I) @ self.phi @ self.u

        self.learnt_parameters.append(self.learnt_parameter)

        return self.learnt_parameter

    def learner_circle(self, inital_q1q2=[0, 0], inital_q1dq2d=[1, 0], plot=True):
        """
        This function moves the robot arm in a circle using the learned torque policy.
        
        Args:
            inital_q1q2: list of two floats, default [0, 0]. The initial joint angles in radians.
            inital_q1dq2d: list of two floats, default [1, 0]. The initial joint velocities in radians/second.
            plot: bool, default True. Whether to plot the robot arm and the trajectory.
        
        Returns:
            None
        """
        # Set initial joint angles and velocities
        self.set_joint_angles(inital_q1q2[0], inital_q1q2[1])
        self.set_joint_velocity(inital_q1dq2d[0], inital_q1dq2d[1])
        
        # Move the robot arm in a circle until it completes a full circle
        while self.joint1_angle - inital_q1q2[0] < 2*np.pi:
            
            q1, q2 = self.joint1_angle, self.joint2_angle
            q1_dot, q2_dot = self.joint1_velocity, self.joint2_velocity

            # Calculate the joint torques required to produce the desired acceleration using the learned parameter
            joint_torques = np.dot(self.learnt_parameter.T, self.calcu_phi(q1, q2, q1_dot, q2_dot))

            # Set the joint torques of the robot object
            self.joint1_torque = joint_torques[0]
            self.joint2_torque = joint_torques[1]
            
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
                plt.title('Student')
                plt.pause(0.001)

            # Update joint states of the robot arm
            self.update_joint_states()

    def learner_pd_controller(self, desired_position, desired_velocity):
        """
        Calculate the joint torques required to produce the desired acceleration using the learned parameter.

        Args:
            desired_position (list): The desired position of the end-effector in the form [x, y].
            desired_velocity (list): The desired velocity of the end-effector in the form [vx, vy].

        Returns:
            tuple: The joint torques required to produce the desired acceleration, delta position, and delta velocity.

        Raises:
            ValueError: If the learner has not learned any parameters.
        """
        # Calculate the current position and velocity of the end-effector
        current_position = self.forward_kinematics()
        current_velocity = np.dot(self.jacobian(), [self.joint1_velocity, self.joint2_velocity])

        # Calculate the errors in position and velocity
        delta_position = np.array(desired_position) - np.array(current_position)
        delta_velocity = np.array(desired_velocity) - np.array(current_velocity)

        # Calculate the joint torques required to produce the desired acceleration
        joint_torques = np.dot(self.learnt_parameter.T, np.concatenate((delta_position, delta_velocity)))

        # Update the joint torques of the robot object and return them
        self.joint1_torque = joint_torques[0]
        self.joint2_torque = joint_torques[1]

        # Update the joint angles of the robot
        self.update_joint_states()
        
        return joint_torques, delta_position, delta_velocity

    def perform_learnt_outcomes(self, desired_position, desired_velocity, threshold=[0.01, 0.5], max_iterations=1000, plot=False):
        """
        Uses the learned parameters to move the robot arm to a desired position using PD control.

        Args:
            desired_position (list): The desired end-effector position [x, y] (in meters).
            desired_velocity (list): The desired end-effector velocity [vx, vy] (in meters/second).
            threshold (list): The position and velocity thresholds for considering the goal as reached (default [0.01, 0.5]).
            max_iterations (int): The maximum number of iterations for the PD control loop (default 1000).
            plot (bool): Whether to plot the trajectory of the robot arm (default False).

        Raises:
            ValueError: If the learner hasn't learnt anything yet.

        Returns:
            tuple: The joint torques required to produce the desired acceleration, the position error and the velocity error.
        """
        if self.learnt_parameter is None:
            raise ValueError('The learner hasn\'t learnt anything.')

        # Extract the desired position coordinates
        x, y = desired_position[0], desired_position[1]

        # Initialize variables
        end_effector_position = np.array([x, y])
        end_effector_velocity = np.array([desired_velocity[0], desired_velocity[1]])
        iteration = 0

        delta_position = np.array([0, 0])
        delta_velocity = np.array([0, 0])

        # PD control loop
        while True:
            # Calculate current end-effector position and velocity
            current_position = self.forward_kinematics()
            current_velocity = np.dot(self.jacobian(), [self.joint1_velocity, self.joint2_velocity])

            # Check if goal is reached
            if np.linalg.norm(end_effector_position - np.array(current_position)) < threshold[0] and np.linalg.norm(end_effector_velocity - np.array(current_velocity)) < threshold[1] or iteration >= max_iterations:
                break

            # Compute desired position and velocity using PD controller
            desired_position = np.array([x, y])
            desired_velocity = desired_velocity

            # Calculate the joint torques required to produce the desired acceleration using the learned parameters
            joint_torques, delta_position_, delta_velocity_ = self.learner_pd_controller(desired_position, desired_velocity)

            # Update the delta position and velocity
            delta_position = delta_position_
            delta_velocity = delta_velocity_

            # Add the current end-effector position to the trajectory
            self.trajectory.append(self.get_end_effector_position())

            # Plot the current state of the robot arm and the trajectory (if requested)
            if plot:
                plt.clf()
                plt.plot(desired_position[0], desired_position[1], '*', c='green')
                plt.xlim(self.x_min, self.x_max)
                plt.ylim(self.y_min, self.y_max)
                plt.plot([0, self.link1_length * np.cos(self.joint1_angle), self.link1_length * np.cos(self.joint1_angle) + self.link2_length * np.cos(self.joint1_angle + self.joint2_angle)], [0, self.link1_length * np.sin(self.joint1_angle), self.link1_length * np.sin(self.joint1_angle) + self.link2_length * np.sin(self.joint1_angle + self.joint2_angle)], '-o')
                plt.plot([pos[0] for pos in self.trajectory], [pos[1] for pos in self.trajectory], '-r')
                plt.grid()
                plt.draw()
                plt.title('Learner')
                plt.pause(0.001)
            
            # Update iteration count
            iteration += 1
        
        # Save RMSE error between learner's final position and desired position
        self.rmse.append(np.sqrt(sum(delta_position)**2 + sum(delta_velocity)**2))

        # Reset states of the robot for the next experiments
        self.reset_robot()
        self.learnt_parameter = None
        
