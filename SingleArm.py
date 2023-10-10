import numpy as np

class SingleRobotArm:
    # properties of this single arm robot 
    def __init__(self, link1_length, link1_mass, joint1_angle, joint1_velocity, joint1_torque, time_step, g):
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
        controller (Controller obj): controller of the robot
        """
        # inital states
        self.joint1_angle0 = joint1_angle   # initial angle of the first joint
        self.joint1_velocity0 = joint1_velocity   # initial angular velocity of the first joint
        self.joint1_torque0 = joint1_torque       # torque applied to the first joint

        # Environmental properties
        self.g = g   # acceleration due to gravity

        # Robot arm properties
        self.link1_length = link1_length   # length of the first arm link
        self.link1_mass = link1_mass       # mass of the first arm link

        # Robot states
        self.joint1_angle = joint1_angle   # initial angle of the first joint
        self.joint1_velocity = joint1_velocity   # initial angular velocity of the first joint
        self.joint1_torque = joint1_torque       # torque applied to the first joint


        # Workspace limitations
        self.x_min = -(link1_length)   # minimum x-coordinate of the workspace
        self.x_max = link1_length    # maximum x-coordinate of the workspace
        self.y_min = -(link1_length)   # minimum y-coordinate of the workspace
        self.y_max = link1_length   # maximum y-coordinate of the workspace

        # Simulation time step
        self.time_step = time_step   # time step for the simulation

        # End-effector trajectory
        self.trajectory = []
        self.velocity = []

        # joint angles his and joint velocity his
        self.joint1_angle_track = []
        self.joint1_velocity_track = []

        # torque track
        self.torque_track = []
        # state track
        self.state_track = []

        # controller gain matrix
        self.L = np.array([[3.0733, 2.6045]])

        self.real_force = []
        self.noise_force = []
        self.phi = []

        # initialize noise level
        self.mean = 0
        self.std = 0
    
    def reset(self):
        # Robot states
        self.joint1_angle = self.joint1_angle0   # initial angle of the first joint
        self.joint1_velocity = self.joint1_velocity0   # initial angular velocity of the first joint
        self.joint1_torque = self.joint1_torque0     # torque applied to the first joint

        # End-effector trajectory
        self.trajectory = []
        self.velocity = []

        # joint angles his and joint velocity his
        self.joint1_angle_track = []
        self.joint1_velocity_track = []


        # torque track
        self.torque_track = []
        # state track
        self.state_track = []

        # controller gain matrix
        self.L = np.array([[3.0733, 2.6045]])

        self.real_force = []
        self.noise_force = []
        self.phi = []

        # initialize noise level
        self.mean = 0
        self.std = 0.1

    def set_L(self, L=np.array([3.0733, 2.6045])):
        self.L = L

    def jacobian(self):
        """
        Computes the Jacobian matrix for the robot arm.
        The Jacobian matrix expresses the relationship between the joint velocities and the end effector velocity.
        The matrix is defined as:
        one-link:
        J = [-L1*sin(theta1), L1*cos(theta1)]
        dx/d(theta1) = -L1*sin(theta1)
        dy/d(theta1) = L1*cos(theta1)
        Two-link:
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
        J = np.array([[-self.link1_length * np.sin(self.joint1_angle)], [self.link1_length * np.cos(self.joint1_angle)]])   # Jacobian matrix with a small regularization term
        # L1 = self.link1_length   # length of the first arm link
        # L2 = self.link2_length   # length of the second arm link
        # J11 = -L1 * np.sin(self.joint1_angle) - L2 * np.sin(self.joint1_angle + self.joint2_angle)   # element (1,1) of the Jacobian matrix
        # J12 = -L2 * np.sin(self.joint1_angle + self.joint2_angle)   # element (1,2) of the Jacobian matrix
        # J21 = L1 * np.cos(self.joint1_angle) + L2 * np.cos(self.joint1_angle + self.joint2_angle)   # element (2,1) of the Jacobian matrix
        # J22 = L2 * np.cos(self.joint1_angle + self.joint2_angle)   # element (2,2) of the Jacobian matrix
        # J = np.array([[J11, J12], [J21, J22]]) + 1e-6 * np.eye(2)   # Jacobian matrix with a small regularization term
        return J

    def jacobian_from_states(self, q):
        """
        Computes the Jacobian matrix for the robot arm from the given states.
        The Jacobian matrix expresses the relationship between the joint velocities and the end effector velocity.
        The matrix is defined as:
        ONE-LINK:
        J = [-L1*sin(theta1), L1*cos(theta1)]
        where
        dx/d(theta1) = -L1*sin(theta1)
        dy/d(theta1) = L1*cos(theta1)
        Two-link:
        J = [dx/d(theta1)  dx/d(theta2)]
            [dy/d(theta1)  dy/d(theta2)]
        where
        dx/d(theta1) = -L1*sin(theta1) - L2*sin(theta1 + theta2)
        dx/d(theta2) = -L2*sin(theta1 + theta2)
        dy/d(theta1) = L1*cos(theta1) + L2*cos(theta1 + theta2)
        dy/d(theta2) = L2*cos(theta1 + theta2)
        Parameters:
        q (array): robot joint angles
        Returns:
        J (array): Jacobian matrix for the robot arm
        """

        L1 = self.link1_length   # length of the first arm link

        joint1_angle = q

        J = np.array([-L1 * np.sin(joint1_angle), L1 * np.cos(joint1_angle)])  # Jacobian matrix with a small regularization term

        # J11 = -L1 * np.sin(joint1_angle) - L2 * np.sin(joint1_angle + joint2_angle)   # element (1,1) of the Jacobian matrix
        # J12 = -L2 * np.sin(joint1_angle + joint2_angle)   # element (1,2) of the Jacobian matrix
        # J21 = L1 * np.cos(joint1_angle) + L2 * np.cos(joint1_angle + joint2_angle)   # element (2,1) of the Jacobian matrix
        # J22 = L2 * np.cos(joint1_angle + joint2_angle)   # element (2,2) of the Jacobian matrix
        # J = np.array([[J11, J12], [J21, J22]]) + 1e-7 * np.eye(2)   # Jacobian matrix with a small regularization term
        return J

    def forward_kinematics(self):
        """
        Computes the end effector position using forward kinematics.
        The forward kinematics equations are:
        One-link:
        x = L1 * cos(theta1)
        y = L1 * sin(theta1)
        Two-link:
        x = L1 * cos(theta1) + L2 * cos(theta1 + theta2)
        y = L1 * sin(theta1) + L2 * sin(theta1 + theta2)
        Returns:
        x (float): x-coordinate of the end effector
        y (float): y-coordinate of the end effector
        """
        x = self.link1_length * np.cos(self.joint1_angle)   # x-coordinate of the end effector
        y = self.link1_length * np.sin(self.joint1_angle)   # y-coordinate of the end effector
        return x, y
    
    def forward_kinematics_from_states(self, q):
        """
        Computes the end effector position using forward kinematics.
        The forward kinematics equations are:
        one-link:
        x = L1 * cos(theta1)
        y = L1 * sin(theta1)
        two-link:
        x = L1 * cos(theta1) + L2 * cos(theta1 + theta2)
        y = L1 * sin(theta1) + L2 * sin(theta1 + theta2)
        Arguments:
        q (array): joint angles
        Returns:
        x (float): x-coordinate of the end effector
        y (float): y-coordinate of the end effector
        """
        joint1_angle = q

        x = self.link1_length * np.cos(joint1_angle)   # x-coordinate of the end effector
        y = self.link1_length * np.sin(joint1_angle)   # y-coordinate of the end effector
        return x, y
        
    def forward_kinematics_from_states_2_joints(self, q):
        x1 = self.link1_length * np.cos(q)
        y1 = self.link1_length * np.sin(q)

        return x1, y1

    def inverse_kinematics_(self, x, y):
        """
        Computes the joint angles required to reach a given end effector position using inverse kinematics.
        The inverse kinematics equations are:
        one-link:
        theta1 = arctan2(y, x)
        two-link:
        theta2 = arccos((x^2 + y^2 - L1^2 - L2^2) / (2 * L1 * L2))
        theta1 = atan2(y, x) - atan2(L2 * sin(theta2), L1 + L2 * cos(theta2))
        Parameters:
        x (float): x-coordinate of the end effector
        y (float): y-coordinate of the end effector
        Returns:
        theta1 (float): angle of the first joint
        theta2 (float): angle of the second joint
        """
        theta1 = np.arctan2(y, x)   # angle of the first joint
        return theta1

    def inverse_kinematics(self, x, y):
        return np.arctan2(y, x)
    
    def motion_dynamics_matrix_MCG_from_states(self, q, q_dot):
        """
        Calculates the inertial matrix, the Coriolis and centrifugal effects vector, and the gravity vector using the given robot state and kinematic parameters.
        Arguments:
            q (numpy array): joint angles
            q_dot (numpy array): joint angular velocity
        Returns:
            M (numpy array): the inertial matrix
            c (numpy array): the Coriolis and centrifugal effects vector
            G (numpy array): the gravity vector
        """
        q1, q2 = q[0], q[1]    # joint angles
        q1_dot, q2_dot = q_dot[0], q_dot[1]    # joint velocities
        m1, m2 = self.link1_mass, self.link2_mass    # masses of the first and second links, respectively
        l1, l2 = self.link1_length, self.link2_length    # lengths of the first and second links, respectively

        M = np.array([[m1*l1*l1 + m2*(l1*l1 + 2*l1*l2*np.cos(q2) + l2*l2), m2*(l1*l2*np.cos(q2) + l2*l2)], 
                      [m2*(l1*l2*np.cos(q2) + l2*l2), m2*l2*l2]])
        
        C = np.array([[-m2*l1*l2*np.sin(q2)*q2_dot, -m2*l1*l2*np.sin(q2)*(q1_dot + q2_dot)], 
                      [-m2*l1*l2*np.sin(q2)*q1_dot, 0]])

        G = np.array([[(m1+m2)*l1*self.g*np.cos(q1) + m2*self.g*l2*np.cos(q1+q2)], [m2*self.g*l2*np.cos(q1+q2)]])

        return M, C, G
    
    def inverse_motion_dynamics_from_states(self, q, q_dot, torque):
        """
        Calculates the joint accelerations required to produce the given torques using the given robot
        state and kinematic parameters.

        q_ddot = inv(M) (torques - Cq_dot - G)
        
        Arguments:
            q (numpy array): joint angles
            q_dot (numpy array): joint angular velocity
            toruqe (numpy array): joint torques
        Returns:
            q1_ddot (float): joint 1 acceleration required to produce the current joint torque
            q2_ddot (float): joint 2 acceleration required to produce the current joint torque
        """
        M, C, G = self.motion_dynamics_matrix_MCG_from_states(q, q_dot)

        q_ddot = np.linalg.inv(M) @ (np.array([[torque[0]], [torque[1]]]) - C @ np.array([[q_dot[0]], [q_dot[1]]]) - G)

        return q_ddot
    
    def controller_from_states(self, q, q_dot, L=np.array([[-1], [-1]])):
        
        torque = L @ np.array([[q], [q_dot]])
        return torque, np.array([[q], [q_dot]])

    def dydt(self, q, q_dot):
        J = self.jacobian_from_states(q)

        M, C, G = self.motion_dynamics_matrix_MCG_from_states(q, q_dot)

        force = self.controller_from_states(q, q_dot, L=self.L)[0]

        torques = J.T @ force + C @ np.array([[q_dot[0]], [q_dot[1]]]) + G
        joint1_torque = torques

        q1_ddot = self.inverse_motion_dynamics_from_states(q, q_dot, joint1_torque)

        return np.array([q_dot, q1_ddot, q2_ddot]), torques

    def update(self):
        """
        Updates the robot arm's state based on the current state and time step.
        Calculates the required joint angles and velocities to achieve the desired end-effector position and velocity.
        """
        q = np.array([self.joint1_angle, self.joint2_angle])
        q_dot = np.array([self.joint1_velocity, self.joint2_velocity])

        real_force, phi = self.controller_from_states(q, q_dot, L=self.L)
        noise_force = real_force + np.random.normal(self.mean,self.std,2).reshape((2,1))

        self.real_force.append(real_force)
        self.noise_force.append(noise_force)
        self.phi.append(phi)

        k1, torques = self.dydt(q, q_dot)
        k2, _ = self.dydt(q + 0.5 * self.time_step * k1[:2], q_dot + 0.5 * self.time_step * k1[2:])
        k3, _ = self.dydt(q + 0.5 * self.time_step * k2[:2], q_dot + 0.5 * self.time_step * k2[2:])
        k4, _ = self.dydt(q + self.time_step * k3[:2], q_dot + self.time_step * k3[2:])

        q_next = q + self.time_step * (k1[:2] + 2 * k2[:2] + 2 * k3[:2] + k4[:2]) / 6
        q_dot_next = q_dot + self.time_step * (k1[2:] + 2 * k2[2:] + 2 * k3[2:] + k4[2:]) / 6

        self.joint1_angle, self.joint2_angle = q_next[0], q_next[1]
        self.joint1_velocity, self.joint2_velocity = q_dot_next[0], q_dot_next[1]

        self.joint1_angle_track.append(self.joint1_angle)
        self.joint2_angle_track.append(self.joint2_angle)
        self.joint1_velocity_track.append(self.joint1_velocity)
        self.joint2_velocity_track.append(self.joint2_velocity)

        self.torque_track.append([torques[0, 0], torques[1,0]])  


    def set_noise_level(self, mean, std):
        self.mean = mean
        self.std = std

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
    
    def get_end_effector_position_from_states(self, q):
        """
        Computes the current position of the end effector.
        Parameters:
        None
        Returns:
        x (float): x-coordinate of the end effector
        y (float): y-coordinate of the end effector
        """
        return self.forward_kinematics_from_states(q)
    
    def get_end_effector_velocity(self):
        """
        Computes the current velocity of the end effector.
        Parameters:
        None
        Returns:
        x_dot (float): x-velocity of the end effector
        y_dot (float): y-velocity of the end effector
        """
        J = self.jacobian() # 1 by 2

        v = J.T * self.joint1_velocity #
        x_dot, y_dot = v[0], v[0]

        return x_dot, y_dot
    
    def get_joint_velocity_from_ee(self, velocity):
        J = self.jacobian() # 2 by 1

        joint_velocity = np.linalg.pinv(J) @ np.array([[velocity[0]], [velocity[1]]])

        return joint_velocity[0]
    
    def get_end_effector_velocity_from_states(self, q, q_dot):

        J = self.jacobian_from_states(q)
        print('shape of J', J.shape)
        print('velocity shape',np.array([q_dot]).shape)
        v = J.T @ np.array([q_dot])
        
        x_dot, y_dot = v[0], v[1]

        return x_dot, y_dot

    def reward_function(self, states, actions):
        Q = self.time_step * np.array([[1, 0], [0, 0.1]])
        R = self.time_step * np.array([[0.1]])
        cost = np.dot(np.dot(np.transpose(states), Q), states) + np.dot(np.dot(np.transpose(actions), R), actions) 
        return -cost

    def states_to_X_si(self, states):
        # Convert states in  xy to polar
        # Convert states to X_si: states (4,6) -> x,y,vx,vy -> theta, theta_dot -> X_si (2,6)
        angular_positions = self.inverse_kinematics_(states[0], states[1])
        angular_velocities = - (states[2] * states[1] - states[3] * states[0]) / self.link1_length
        return np.vstack((angular_positions.reshape(1, -1), angular_velocities))
    
    def actions_to_joint_torques(self, states, actions):
        theta = np.arctan2(states[1], states[0])

        # Step 2: Compute the unit vectors along the pendulum rod for each instance
        r_unit = np.array([np.cos(theta), np.sin(theta)])

        # Step 3: Compute the perpendicular components of the applied forces
        # The cross product in 2D simplifies to a scalar value: fx * ry - fy * rx
        F_perp = actions[0] * r_unit[1] - actions[1] * r_unit[0]
        return F_perp.reshape(1, -1)