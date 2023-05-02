import numpy as np

class BasicRobotArm:
    def __init__(self, 
                 link1_length, link2_length, link1_mass, link2_mass, 
                 joint1_angle, joint2_angle, joint1_velocity, joint2_velocity, joint1_torque, joint2_torque,
                 time_step, g):
        """
        Initializes the robot arm with its physical properties and initial states.

        Parameters:
        link1_length (float): length of the first arm link
        link2_length (float): length of the second arm link
        link1_mass (float): mass of the first arm link
        link2_mass (float): mass of the second arm link
        joint1_angle (float): initial angle of the first joint
        joint2_angle (float): initial angle of the second joint
        joint1_velocity (float): initial angular velocity of the first joint
        joint2_velocity (float): initial angular velocity of the second joint
        joint1_torque (float): torque applied to the first joint
        joint2_torque (float): torque applied to the second joint
        time_step (float): time step for the simulation
        g (float): acceleration due to gravity
        """
        # Robot arm properties
        self.link1_length = link1_length   # length of the first arm link
        self.link2_length = link2_length   # length of the second arm link
        self.link1_mass = link1_mass       # mass of the first arm link
        self.link2_mass = link2_mass       # mass of the second arm link

        # Robot states
        self.joint1_angle = joint1_angle   # initial angle of the first joint
        self.joint2_angle = joint2_angle   # initial angle of the second joint
        self.joint1_velocity = joint1_velocity   # initial angular velocity of the first joint
        self.joint2_velocity = joint2_velocity   # initial angular velocity of the second joint
        self.joint1_torque = joint1_torque       # torque applied to the first joint
        self.joint2_torque = joint2_torque       # torque applied to the second joint

        # Environmental properties
        self.g = g   # acceleration due to gravity

        # Moment of inertia
        self.I1 = (1/12)*self.link1_mass*self.link1_length*self.link1_length   # moment of inertia of the first link
        self.I2 = (1/12)*self.link2_mass*self.link2_length*self.link2_length   # moment of inertia of the second link

        # Workspace limitations
        self.x_min = -(link1_length + link2_length)   # minimum x-coordinate of the workspace
        self.x_max = link1_length + link2_length      # maximum x-coordinate of the workspace
        self.y_min = -(link1_length + link2_length)   # minimum y-coordinate of the workspace
        self.y_max = link1_length + link2_length      # maximum y-coordinate of the workspace

        # Simulation time step
        self.time_step = time_step   # time step for the simulation

    def forward_kinematics(self):
        """
        Computes the end effector position using forward kinematics.

        The forward kinematics equations are:

        x = L1 * cos(theta1) + L2 * cos(theta1 + theta2)
        y = L1 * sin(theta1) + L2 * sin(theta1 + theta2)

        Returns:
        x (float): x-coordinate of the end effector
        y (float): y-coordinate of the end effector
        """
        x = self.link1_length * np.cos(self.joint1_angle) + \
            self.link2_length * np.cos(self.joint1_angle + self.joint2_angle)   # x-coordinate of the end effector
        y = self.link1_length * np.sin(self.joint1_angle) + \
            self.link2_length * np.sin(self.joint1_angle + self.joint2_angle)   # y-coordinate of the end effector
        return x, y

        
    def inverse_kinematics(self, x, y):
        """
        Computes the joint angles required to reach a given end effector position using inverse kinematics.

        The inverse kinematics equations are:

        theta2 = arccos((x^2 + y^2 - L1^2 - L2^2) / (2 * L1 * L2))
        theta1 = atan2(y, x) - atan2(L2 * sin(theta2), L1 + L2 * cos(theta2))

        Parameters:
        x (float): x-coordinate of the end effector
        y (float): y-coordinate of the end effector

        Returns:
        theta1 (float): angle of the first joint
        theta2 (float): angle of the second joint
        """
        L1 = self.link1_length   # length of the first arm link
        L2 = self.link2_length   # length of the second arm link
        #theta2 = np.arccos((x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2))   # angle of the second joint
        #theta1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(theta2), L1 + L2 * np.cos(theta2))   # angle of the first joint
        theta2 = np.arccos((x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2))
        theta1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(theta2), (L1 + L2 * np.cos(theta2)) )
        #theta1 = np.arctan2(y, x) - np.arcsin(L2 * np.sin(theta2) / (np.sqrt(x**2 + y**2)))
        return theta1, theta2
    
    def calcu_phi(self, q1, q2, q1_dot, q2_dot):
        return np.array([np.cos(q1), np.cos(q1+q2), q1_dot*q1_dot*np.sin(q2), (2*q1_dot*q2_dot+q2_dot*q2_dot)*np.sin(q2)])
    
    def motion_of_dynamics_matrix(self, bigC=True):
        """
        Calculates the mass matrix, the Coriolis and centrifugal effects vector, and the gravity vector using the robot
        arm's current state and kinematic parameters.
        Returns:
            M (numpy array): the mass matrix
            c (numpy array): the Coriolis and centrifugal effects vector
            G (numpy array): the gravity vector
        """
        q1, q2 = self.joint1_angle, self.joint2_angle    # current joint angles
        q1_dot, q2_dot = self.joint1_velocity, self.joint2_velocity    # current joint velocities
        m1, m2 = self.link1_mass, self.link2_mass    # masses of the first and second links, respectively
        l1, l2 = self.link1_length, self.link2_length    # lengths of the first and second links, respectively

        M = np.array([[m1*l1*l1 + m2*(l1*l1 + 2*l1*l2*np.cos(q2) + l2*l2), m2*(l1*l2*np.cos(q2) + l2*l2)], 
                      [m2*(l1*l2*np.cos(q2) + l2*l2), m2*l2*l2]])
        
        c = np.array([[-m2*l1*l2*np.sin(q2)*(2*q1_dot*q2_dot + q2_dot*q2_dot)], [m2*l1*l2*q1_dot*q1_dot*np.sin(q2)]])
        C = np.array([[-2*m2*l1*l2*np.sin(q2)*q2_dot, -m2*l1*l2*np.sin(q2)*q2_dot],[m2*l1*l2*q1_dot*np.sin(q2), 0]])
        
        G = np.array([[(m1+m2)*l1*self.g*np.cos(q1) + m2*self.g*l2*np.cos(q1+q2)], [m2*self.g*l2*np.cos(q1+q2)]])
        
        if bigC:
            return M, C, G
        else:
            return M, c, G
    
    def motion_of_dynamics_torque(self, q1_ddot, q2_ddot):
        """
        Calculates the joint torques required to produce the given joint accelerations using the robot arm's current state
        and kinematic parameters.
        Args:
            q1_ddot (float): desired acceleration of joint 1
            q2_ddot (float): desired acceleration of joint 2
        Returns:
            joint1_torque (float): torque required to produce the desired acceleration of joint 1
            joint2_torque (float): torque required to produce the desired acceleration of joint 2
        """
        M, c, G = self.motion_of_dynamics_matrix()

        torques = M @ np.array([[q1_ddot], [q2_ddot]]) + c

        return torques[0, 0], torques[1, 0]
    
    def motion_of_dynamics_qddot(self):
        """
        Calculates the joint accelerations required to produce the current joint torques using the robot arm's current
        state and kinematic parameters.

        q_ddot = inv(M) (torques - c - G)

        Returns:
            q1_ddot (float): joint 1 acceleration required to produce the current joint torque
            q2_ddot (float): joint 2 acceleration required to produce the current joint torque
        """
        M, c, G = self.motion_of_dynamics_matrix()

        q_ddot = np.linalg.inv(M) @ (np.array([[self.joint1_torque], [self.joint2_torque]]) - c)
        #q_ddot = np.linalg.inv(M) @ (np.array([[self.joint1_torque], [self.joint2_torque]]))

        return q_ddot[0, 0], q_ddot[1, 0]
    
    def ee_motion_of_dynamics_matrix(self):
        J = self.jacobian()

        M, C, G = self.motion_of_dynamics_matrix(bigC=True)

        M_ee = np.linalg.inv(J).T @ M @ np.linalg.inv(J)
        C_ee = np.linalg.inv(J).T @ C @ np.linalg.inv(J) - Lambda @ J @ np.linalg.inv(J)
        G_ee = np.linalg.inv(J).T @ G

        return M_ee, C_ee, G_ee
    
    def ee_motion_of_dynamics_force(self, x_dot, x_ddot):
        M_ee, C_ee, G_ee = self.ee_motion_of_dynamics_matrix()

        return M_ee @ x_ddot + C_ee @ x_dot + G_ee
    
    def ee_motion_of_dynamics_xddot(self, x_dot, force):
        M_ee, C_ee, G_ee = self.ee_motion_of_dynamics_matrix()

        return np.linalg.inv(M_ee) @ (force - C_ee @ x_dot - G_ee)
    
    def jacobian(self):
        """
        Computes the Jacobian matrix for the robot arm.

        The Jacobian matrix expresses the relationship between the joint velocities and the end effector velocity.
        The matrix is defined as:

        J = [dx/d(theta1)  dx/d(theta2)]
            [dy/d(theta1)  dy/d(theta2)]

        where
        dx/d(theta1) = -L1*sin(theta1) - L2*sin(theta1 + theta2)
        dx/d(theta2) = -L2*sin(theta1 + theta2)
        dy/d(theta1) = L1*cos(theta1) + L2*cos(theta1 + theta2)
        dy/d(theta2) = L2*cos(theta1 + theta2)

        Parameters:
        None

        Returns:
        J (array): Jacobian matrix for the robot arm
        """
        L1 = self.link1_length   # length of the first arm link
        L2 = self.link2_length   # length of the second arm link
        J11 = -L1 * np.sin(self.joint1_angle) - L2 * np.sin(self.joint1_angle + self.joint2_angle)   # element (1,1) of the Jacobian matrix
        J12 = -L2 * np.sin(self.joint1_angle + self.joint2_angle)   # element (1,2) of the Jacobian matrix
        J21 = L1 * np.cos(self.joint1_angle) + L2 * np.cos(self.joint1_angle + self.joint2_angle)   # element (2,1) of the Jacobian matrix
        J22 = L2 * np.cos(self.joint1_angle + self.joint2_angle)   # element (2,2) of the Jacobian matrix
        J = np.array([[J11, J12], [J21, J22]]) + 1e-6 * np.eye(2)   # Jacobian matrix with a small regularization term
        return J
    
    def update_joint_states(self):
        """
        Updates the joint positions and velocities based on the applied torques and time step.

        This function uses the current joint positions, velocities, and applied torques to calculate the
        new joint positions and velocities after a given time step, using the following equations of motion:
            q_new = q + q_dot*dt + 0.5*q_dot_dot*dt^2
            q_dot_new = q_dot + q_dot_dot*dt

        The calculated joint positions and velocities are then set as the new joint states for the robot arm.

        Parameters:
        None

        Returns:
        None
        """
        # Get the current joint angles and velocities
        q1, q2 = self.get_joint_angles()

        # Compute the joint accelerations based on the applied torques and moment of inertia
        q1_dot_dot, q2_dot_dot = self.motion_of_dynamics_qddot()

        # Calculate the new joint positions and velocities based on the equations of motion
        q1_new = q1 + self.joint1_velocity * self.time_step + 0.5 * q1_dot_dot * self.time_step * self.time_step
        q2_new = q2 + self.joint2_velocity * self.time_step + 0.5 * q2_dot_dot * self.time_step * self.time_step

        q1_dot_new = self.joint1_velocity + q1_dot_dot * self.time_step
        q2_dot_new = self.joint2_velocity + q2_dot_dot * self.time_step

        # Set the new joint angles and velocities
        self.set_joint_angles(q1_new, q2_new)
        self.set_joint_velocity(q1_dot_new, q2_dot_new)
        

    def set_joint_torques(self, joint1_torque, joint2_torque):
        """
        Sets the torques applied to each joint.

        Parameters:
        joint1_torque (float): torque applied to the first joint
        joint2_torque (float): torque applied to the second joint

        Returns:
        None
        """
        self.joint1_torque = joint1_torque
        self.joint2_torque = joint2_torque

    def get_joint_torques(self):
        """
        Gets the torques currently applied to each joint.

        Parameters:
        None

        Returns:
        joint1_torque (float): torque currently applied to the first joint
        joint2_torque (float): torque currently applied to the second joint
        """
        return self.joint1_torque, self.joint2_torque

    def set_joint_velocity(self, joint1_velocity, joint2_velocity):
        """
        Sets the angular velocity of each joint.

        Parameters:
        joint1_velocity (float): angular velocity of the first joint
        joint2_velocity (float): angular velocity of the second joint

        Returns:
        None
        """
        self.joint1_velocity = joint1_velocity
        self.joint2_velocity = joint2_velocity

    def set_joint_angles(self, joint1_angle, joint2_angle):
        """
        Sets the angles of each joint.

        Parameters:
        joint1_angle (float): angle of the first joint
        joint2_angle (float): angle of the second joint

        Returns:
        None
        """
        self.joint1_angle = joint1_angle
        self.joint2_angle = joint2_angle

    def get_joint_angles(self):
        """
        Gets the current angles of each joint.

        Parameters:
        None

        Returns:
        joint1_angle (float): current angle of the first joint
        joint2_angle (float): current angle of the second joint
        """
        return self.joint1_angle, self.joint2_angle

    def get_end_effector_position(self):
        """
        Computes the current position of the end effector.

        Parameters:
        None

        Returns:
        x (float): x-coordinate of the end effector
        y (float): y-coordinate of the end effector
        """
        return self.forward_kinematics()

    def get_link_masses(self):
        """
        Gets the masses of each link.

        Parameters:
        None

        Returns:
        link1_mass (float): mass of the first link
        link2_mass (float): mass of the second link
        """
        return self.link1_mass, self.link2_mass

