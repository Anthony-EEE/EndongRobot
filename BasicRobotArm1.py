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
        controller (Controller obj): controller of the robot
        """
        # inital states
        self.joint1_angle0 = np.float32(joint1_angle)   # initial angle of the first joint
        self.joint2_angle0 = np.float32(joint2_angle)   # initial angle of the second joint
        self.joint1_velocity0 = np.float32(joint1_velocity)   # initial angular velocity of the first joint
        self.joint2_velocity0 = np.float32(joint2_velocity)   # initial angular velocity of the second joint
        self.joint1_torque0 = np.float32(joint1_torque)       # torque applied to the first joint
        self.joint2_torque0 = np.float64(joint2_torque)       # torque applied to the second joint

        # Environmental properties
        self.g = np.float64(g)   # acceleration due to gravity

        # Robot arm properties
        self.link1_length = np.float64(link1_length)   # length of the first arm link
        self.link2_length = np.float64(link2_length)   # length of the second arm link
        self.link1_mass = np.float64(link1_mass)       # mass of the first arm link
        self.link2_mass = np.float64(link2_mass)       # mass of the second arm link

        # Robot states
        self.joint1_angle = np.float64(joint1_angle)   # initial angle of the first joint
        self.joint2_angle = np.float64(joint2_angle)   # initial angle of the second joint
        self.joint1_velocity = np.float64(joint1_velocity)   # initial angular velocity of the first joint
        self.joint2_velocity = np.float64(joint2_velocity)   # initial angular velocity of the second joint
        self.joint1_torque = np.float64(joint1_torque)       # torque applied to the first joint
        self.joint2_torque = np.float64(joint2_torque)       # torque applied to the second joint

        # Environmental properties
        self.g = g   # acceleration due to gravity

        # Workspace limitations
        self.x_min = -(link1_length + link2_length)   # minimum x-coordinate of the workspace
        self.x_max = link1_length + link2_length      # maximum x-coordinate of the workspace
        self.y_min = -(link1_length + link2_length)   # minimum y-coordinate of the workspace
        self.y_max = link1_length + link2_length      # maximum y-coordinate of the workspace

        # Simulation time step
        self.time_step = np.float32(time_step)   # time step for the simulation

        # End-effector trajectory
        self.trajectory = []
        self.velocity = []

        # joint angles his and joint velocity his
        self.joint1_angle_track = []
        self.joint2_angle_track = []
        self.joint1_velocity_track = []
        self.joint2_velocity_track = []

        # torque track
        self.torque_track = []
        # state track
        self.state_track = []

        # controller gain matrix
        self.skill_theta = np.array([[1, 0, -1, 0, 0], [0, 1, 0, -1, 0]])

        self.real_force = []
        self.noise_force = []
        self.phi = []

        # initialize noise level
        self.mean = 0
        self.std = 0.1

        self.force_for_test = []

    def reset(self):
        # Robot states
        self.joint1_angle = self.joint1_angle0   # initial angle of the first joint
        self.joint2_angle = self.joint2_angle0   # initial angle of the second joint
        self.joint1_velocity = self.joint1_velocity0   # initial angular velocity of the first joint
        self.joint2_velocity = self.joint2_velocity0   # initial angular velocity of the second joint
        self.joint1_torque = self.joint1_torque0     # torque applied to the first joint
        self.joint2_torque = self.joint2_torque0       # torque applied to the second joint

        # End-effector trajectory
        self.trajectory = []
        self.velocity = []

        # joint angles his and joint velocity his
        self.joint1_angle_track = []
        self.joint2_angle_track = []
        self.joint1_velocity_track = []
        self.joint2_velocity_track = []

        # torque track
        self.torque_track = []
        # state track
        self.state_track = []

        # controller gain matrix
        self.skill_theta = np.array([[1, 0, -1, 0, 0], [0, 1, 0, -1, 0]])

        self.real_force = []
        self.noise_force = []
        self.phi = []

        # initialize noise level
        self.mean = 0
        self.std = 0.1

    def set_skill_theta(self, skill_theta=np.array([[-1, 0, -1, 0, 0], [0, -1, 0, -1, 0]])):
        self.skill_theta = skill_theta

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

    def jacobian_from_states(self, q):
        """
        Computes the Jacobian matrix for the robot arm from the given states.

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
        q (array): robot joint angles

        Returns:
        J (array): Jacobian matrix for the robot arm
        """
        L1 = self.link1_length   # length of the first arm link
        L2 = self.link2_length   # length of the second arm link

        joint1_angle, joint2_angle = q[0], q[1]

        J11 = -L1 * np.sin(joint1_angle) - L2 * np.sin(joint1_angle + joint2_angle)   # element (1,1) of the Jacobian matrix
        J12 = -L2 * np.sin(joint1_angle + joint2_angle)   # element (1,2) of the Jacobian matrix
        J21 = L1 * np.cos(joint1_angle) + L2 * np.cos(joint1_angle + joint2_angle)   # element (2,1) of the Jacobian matrix
        J22 = L2 * np.cos(joint1_angle + joint2_angle)   # element (2,2) of the Jacobian matrix
        J = np.array([[J11, J12], [J21, J22]]) + 1e-7 * np.eye(2)   # Jacobian matrix with a small regularization term
        return J

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
        
        # print("xxxx", x)
        # print("yyyy", y)
        # print("self.joint1_angle", self.joint1_angle)
        # print("self.joint2_angle", self.joint2_angle)

        return x, y
    
    def forward_kinematics_from_states(self, q):
        """
        Computes the end effector position using forward kinematics.

        The forward kinematics equations are:

        x = L1 * cos(theta1) + L2 * cos(theta1 + theta2)
        y = L1 * sin(theta1) + L2 * sin(theta1 + theta2)

        Arguments:
        q (array): joint angles
        Returns:
        x (float): x-coordinate of the end effector
        y (float): y-coordinate of the end effector
        """
        joint1_angle, joint2_angle = q[0], q[1]

        x = self.link1_length * np.cos(joint1_angle) + \
            self.link2_length * np.cos(joint1_angle + joint2_angle)   # x-coordinate of the end effector
        y = self.link1_length * np.sin(joint1_angle) + \
            self.link2_length * np.sin(joint1_angle + joint2_angle)   # y-coordinate of the end effector
        return x, y
        
    def forward_kinematics_from_states_2_joints(self, q):
        q1, q2 = q[0], q[1]
        x1 = self.link1_length * np.cos(q1)
        y1 = self.link1_length * np.sin(q1)
        x2 = x1 + self.link2_length * np.cos(q1+q2)
        y2 = y1 + self.link2_length * np.sin(q1+q2)

        return [x1, x2], [y1, y2]

    def inverse_kinematics_(self, x, y):
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
        L1, L2 = self.link1_length, self.link2_length
        d_sq = x**2 + y**2

        cos_theta2 = (d_sq - L1**2 - L2**2) / (2 * L1 * L2)
        cos_theta2 = np.clip(cos_theta2, -1, 1)
        theta2_1 = np.arccos(cos_theta2)
        theta2_2 = -np.arccos(cos_theta2)

        theta1_1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(theta2_1), L1 + L2 * np.cos(theta2_1))
        theta1_2 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(theta2_2), L1 + L2 * np.cos(theta2_2))

        # Choose the solution with the smallest change in joint angles
        delta_theta1_1 = np.abs(self.joint1_angle - theta1_1)
        delta_theta1_2 = np.abs(self.joint1_angle - theta1_2)

        if delta_theta1_1 < delta_theta1_2:
            return theta1_1, theta2_1
        else:
            return theta1_2, theta2_2

    def inverse_kinematics(self, x, y):
        
        L1, L2 = self.link1_length, self.link2_length
        d_sq = x**2 + y**2

        cos_theta2 = (d_sq - L1**2 - L2**2) / (2 * L1 * L2)
        cos_theta2 = np.clip(cos_theta2, -1, 1)
        theta2 = np.arccos(cos_theta2)

        theta1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(theta2), L1 + L2 * np.cos(theta2))

        return theta1, theta2

    def motion_dynamics_matrix_MCG(self):
        """
        Calculates the inertial matrix, the Coriolis and centrifugal effects vector, and the gravity vector using the robot
        arm's current state and kinematic parameters.
        Returns:
            M (numpy array): the inertial matrix
            c (numpy array): the Coriolis and centrifugal effects vector
            G (numpy array): the gravity vector
        """
        q1, q2 = self.joint1_angle, self.joint2_angle    # current joint angles
        q1_dot, q2_dot = self.joint1_velocity, self.joint2_velocity    # current joint velocities
        m1, m2 = self.link1_mass, self.link2_mass    # masses of the first and second links, respectively
        l1, l2 = self.link1_length, self.link2_length    # lengths of the first and second links, respectively

        M = np.array([[m1*l1*l1 + m2*(l1*l1 + 2*l1*l2*np.cos(q2) + l2*l2), m2*(l1*l2*np.cos(q2) + l2*l2)], 
                      [m2*(l1*l2*np.cos(q2) + l2*l2), m2*l2*l2]])
        
        C = np.array([[-m2*l1*l2*np.sin(q2)*q2_dot, -m2*l1*l2*np.sin(q2)*(q1_dot + q2_dot)], 
                      [-m2*l1*l2*np.sin(q2)*q1_dot, 0]])

        G = np.array([[(m1+m2)*l1*self.g*np.cos(q1) + m2*self.g*l2*np.cos(q1+q2)], [m2*self.g*l2*np.cos(q1+q2)]])

        return M, C, G
    
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
        
        # C = np.array([[-m2*l1*l2*np.sin(q2)*q2_dot, -m2*l1*l2*np.sin(q2)*(q1_dot + q2_dot)], 
        #               [-m2*l1*l2*np.sin(q2)*q1_dot, 0]])

        C = np.array([[-m2*l1*l2*(2*q1_dot*q2_dot + q2_dot*q2_dot)*np.sin(q2)],
                      [m2*l1*l2*q1_dot*q1_dot*np.sin(q2)]])

        G = np.array([[(m1+m2)*l1*self.g*np.cos(q1) + m2*self.g*l2*np.cos(q1+q2)], [m2*self.g*l2*np.cos(q1+q2)]])

        return M, C, G
    
    def motion_dynamics_matrix_os(self):
        """
        Calculates the pesudo inertial matrix, the Coriolis and centrifugal effects vector, 
        and the gravity vector using the robot arm's current state and kinematic parameters. 
        These are for operational space motion dynamics.
        Returns:
            pesudo_M (numpy array): the pesudo inertial matrix
            pesudo_C (numpy array): the pesudo Coriolis and centrifugal effects vector
            pesudo_G (numpy array): the pesudo gravity vector
        """
        # Get the inertia, Coriolis/centrifugal, and gravity matrices using the current robot arm state and kinematic parameters
        M, C, G = self.motion_dynamics_matrix_MCG()
        
        # Get the Jacobian matrix
        J = self.jacobian()

        # Calculate the pesudo-inertial matrix using the Jacobian and the inertia matrix
        pesudo_M = np.linalg.inv(J).T @ M @ np.linalg.inv(J)
        
        # Calculate the pesudo Coriolis and centrifugal effects vector using the Jacobian, 
        # the Coriolis/centrifugal matrix, and the pesudo-inertial matrix
        pesudo_C = np.linalg.inv(J).T @ C @ np.linalg.inv(J) - pesudo_M @ J @ np.linalg.inv(J)
        
        # Calculate the pesudo gravity vector using the Jacobian and the gravity matrix
        pesudo_G = np.linalg.inv(J).T @ G

        # Return the pesudo-inertial matrix, pesudo Coriolis and centrifugal effects vector, and pesudo gravity vector
        return pesudo_M, pesudo_C, pesudo_G
        
    def forward_motion_dynamics(self, q1_ddot, q2_ddot):
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
        M, C, G = self.motion_dynamics_matrix_MCG()

        torques = M @ np.array([[q1_ddot], [q2_ddot]]) + C @ np.array([[self.joint1_velocity], [self.joint2_velocity]]) + G

        return torques[0, 0], torques[1, 0]
        
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

        # q_ddot = np.linalg.inv(M) @ (np.array([[torque[0]], [torque[1]]]) - C @ np.array([[q_dot[0]], [q_dot[1]]]) - G)

        q_ddot = np.linalg.inv(M) @ (np.array([[torque[0]], [torque[1]]]) - C)

        return q_ddot[0, 0], q_ddot[1, 0]
    
    def controller_from_states(self, q, q_dot, skill_theta=np.array([[-1, 0, -1, 0, 0], [0, -1, 0, -1, 0]])):
        """
        Calculates the desired force to control the robot arm's end-effector position and velocity 
        using a linear feedback controller. The controller gain matrix can be customized by providing L.
        Arguments:
            q (numpy array): joint angles
            q_dot (numpy array): joint velocity
            L (numpy array, optional): the controller gain matrix
        Returns:
            force (numpy array): the desired force on the end-effector
        """
        # Get the current end-effector position and velocity
        x, y = self.get_end_effector_position_from_states(q)
        x_dot, y_dot = self.get_end_effector_velocity_from_states(q, q_dot)

        # State track
        self.state_track.append([x, y, x_dot, y_dot])

        # Calculate the desired force using the controller gain matrix and the current end-effector position and velocity
        force = skill_theta @ np.array([[x], [y], [x_dot], [y_dot], [1]])
        self.force_for_test.append(force)
        # print("force", force)
        return force, np.array([[x], [y], [x_dot], [y_dot], [1]])
    
   
    def dydt(self, q, q_dot):
        J = self.jacobian_from_states(q)

        # M, C, G = self.motion_dynamics_matrix_MCG_from_states(q, q_dot)

        force = self.controller_from_states(q, q_dot, skill_theta=self.skill_theta)[0]
        # force[0, 0], force[1, 0] = round(force[0, 0], 3), round(force[1, 0], 3)
        

        # torques = J.T @ force + C @ np.array([[q_dot[0]], [q_dot[1]]]) + G
        torques = J.T @ force
        
        joint1_torque, joint2_torque = torques[0, 0], torques[1, 0]

        q1_ddot, q2_ddot = self.inverse_motion_dynamics_from_states(q, q_dot, [joint1_torque, joint2_torque])

        return np.array([q_dot[0], q_dot[1], q1_ddot, q2_ddot]), torques

    def update_rk4(self):
        """
        Updates the robot arm's state based on the current state and time step.
        Calculates the required joint angles and velocities to achieve the desired end-effector position and velocity.
        """
        q = np.array([self.joint1_angle, self.joint2_angle])
        q_dot = np.array([self.joint1_velocity, self.joint2_velocity])

        real_force, phi = self.controller_from_states(q, q_dot, skill_theta=self.skill_theta)
        # print("real_force", real_force)
        # noise_force = real_force + np.random.normal(self.mean,self.std,2).reshape((2,1))

        self.real_force.append(real_force)
        # self.noise_force.append(noise_force)
        self.phi.append(phi)

        k1, torques = self.dydt(q, q_dot)
        k2, _ = self.dydt(q + 0.5 * self.time_step * k1[:2], q_dot + 0.5 * self.time_step * k1[2:])
        k3, _ = self.dydt(q + 0.5 * self.time_step * k2[:2], q_dot + 0.5 * self.time_step * k2[2:])
        k4, _ = self.dydt(q + self.time_step * k3[:2], q_dot + self.time_step * k3[2:])

        q_next = q + self.time_step * (k1[:2] + 2 * k2[:2] + 2 * k3[:2] + k4[:2]) / 6
        q_dot_next = q_dot + self.time_step * (k1[2:] + 2 * k2[2:] + 2 * k3[2:] + k4[2:]) / 6

        # q_next[0], q_next[1] = round(q_next[0], 3), round(q_next[1], 3)
        # q_dot_next[0], q_dot_next[1] = round(q_dot_next[0], 3), round(q_dot_next[1], 3)

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
        J = self.jacobian()

        v = J @ np.array([[self.joint1_velocity], [self.joint2_velocity]])

        x_dot, y_dot = v[0, 0], v[1, 0]

        return x_dot, y_dot
    
    def get_joint_velocity_from_ee(self, velocity):
        J = self.jacobian()

        joint_velocity = np.linalg.inv(J) @ np.array([[velocity[0]], [velocity[1]]])

        return joint_velocity[0, 0], joint_velocity[1, 0]
    
    def get_end_effector_velocity_from_states(self, q, q_dot):
        """
        Computes the current velocity of the end effector.

        Parameters:
        None

        Returns:
        x_dot (float): x-velocity of the end effector
        y_dot (float): y-velocity of the end effector
        """
        J = self.jacobian_from_states(q)

        v = J @ np.array([[q_dot[0]], [q_dot[1]]])

        x_dot, y_dot = v[0, 0], v[1, 0]

        return x_dot, y_dot