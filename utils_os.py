import random
import itertools

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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

        # Workspace limitations
        self.x_min = -(link1_length + link2_length)   # minimum x-coordinate of the workspace
        self.x_max = link1_length + link2_length      # maximum x-coordinate of the workspace
        self.y_min = -(link1_length + link2_length)   # minimum y-coordinate of the workspace
        self.y_max = link1_length + link2_length      # maximum y-coordinate of the workspace

        # Simulation time step
        self.time_step = time_step   # time step for the simulation

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
        self.L = np.array([[1, 0, -1, 0], [0, 1, 0, -1]])

    def set_L(self, L=np.array([[-1, 0, -1, 0], [0, -1, 0, -1]])):
        self.L = L

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
        theta2 = np.arccos((x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2))   # angle of the second joint
        theta1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(theta2), L1 + L2 * np.cos(theta2))   # angle of the first joint
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
        
    def inverse_motion_dynamics(self):
        """
        Calculates the joint accelerations required to produce the current joint torques using the robot arm's current
        state and kinematic parameters.

        q_ddot = inv(M) (torques - c - G)

        Returns:
            q1_ddot (float): joint 1 acceleration required to produce the current joint torque
            q2_ddot (float): joint 2 acceleration required to produce the current joint torque
        """
        M, C, G = self.motion_dynamics_matrix_MCG()

        q_ddot = np.linalg.inv(M) @ (np.array([[self.joint1_torque], [self.joint2_torque]]) - C @ np.array([[self.joint1_velocity], [self.joint2_velocity]]) - G)

        return q_ddot[0, 0], q_ddot[1, 0]
    
    def controller(self, L=np.array([[-1, 0, -1, 0], [0, -1, 0, -1]])):
        """
        Calculates the desired force to control the robot arm's end-effector position and velocity 
        using a linear feedback controller. The controller gain matrix can be customized by providing L.
        Arguments:
            L (numpy array, optional): the controller gain matrix
        Returns:
            force (numpy array): the desired force on the end-effector
        """
        # Get the pesudo-inertial matrix, pesudo Coriolis and centrifugal effects vector, and pesudo gravity vector
        M, C, G = self.motion_dynamics_matrix_os()

        # Get the current end-effector position and velocity
        x, y = self.get_end_effector_position()
        x_dot, y_dot = self.get_end_effector_velocity()

        # State track
        self.state_track.append([x, y, x_dot, y_dot])

        # Calculate the desired force using the controller gain matrix and the current end-effector position and velocity
        force = L @ np.array([[x], [y], [x_dot], [y_dot]])

        # Return the desired force
        return force

    def update(self):
        """
        Updates the robot arm's state based on the current state and time step.
        Calculates the required joint angles and velocities to achieve the desired end-effector position and velocity.
        """
        # Get the Jacobian matrix
        J = self.jacobian()
        # motion dynamics matrix
        M, C, G = self.motion_dynamics_matrix_MCG()

        # Get the desired force using the linear feedback controller
        force = self.controller(L=self.L)
        # Get the desired toruqes
        torques = J.T @ force + C @ np.array([[self.joint1_velocity], [self.joint2_velocity]]) + G
        self.joint1_torque, self.joint2_torque = torques[0, 0], torques[1,0]
        
        # Calculate the required end-effector acceleration to achieve the desired force
        q1_ddot, q2_ddot = self.inverse_motion_dynamics()

        # Calculate the new joint positions and velocities based on the equations of motion
        self.joint1_angle = self.joint1_angle + self.joint1_velocity * self.time_step + 0.5 * q1_ddot * self.time_step * self.time_step
        self.joint2_angle = self.joint2_angle + self.joint2_velocity * self.time_step + 0.5 * q2_ddot * self.time_step * self.time_step

        self.joint1_velocity = self.joint1_velocity + q1_ddot * self.time_step
        self.joint2_velocity = self.joint2_velocity + q2_ddot * self.time_step

        self.joint1_angle_track.append(self.joint1_angle)
        self.joint2_angle_track.append(self.joint2_angle)
        self.joint1_velocity_track.append(self.joint1_velocity)
        self.joint2_velocity_track.append(self.joint2_velocity)

        self.torque_track.append([torques[0, 0], torques[1,0]])

    def train_and_evaluate_learner(self, model, x, y, train_indices, random_state=None):
        """
        Trains and evaluates a regression model using scikit-learn.

        Args:
            model: A scikit-learn regression model object.
            X: A numpy array or pandas DataFrame of shape (n_samples, n_features).
            y: A numpy array or pandas DataFrame of shape (n_samples, n_outputs).
            train_samples: An integer representing the number of samples to include in the train set.
            random_state: An integer or None, optional (default=None). Controls the random seed.

        Returns:
            A dictionary containing the trained model, and evaluation metrics (mean squared error and R2 score).
        """

        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

        # Create a list of indices for all samples
        indices = list(range(len(x)))

        # Remove the test set indices to get the training set indices
        test_indices = [index for index in indices if index not in train_indices]

        # Split the data into training and testing sets
        X_train, X_test = x[train_indices], x[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Return the trained model and evaluation metrics
        result = {
            'mean_squared_error': mse,
            'root_mse': rmse,
            'r2_score': r2
        }

        return result

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
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    robot = BasicRobotArm(
        link1_length=1, link2_length=1, 
        link1_mass=1, link2_mass=1, 
        joint1_angle=0, joint2_angle=np.pi/5, 
        joint1_velocity=0.0, joint2_velocity=0.0,
        joint1_torque=0.0, joint2_torque=0.0,
        time_step=0.1, g=9.81)
    
    # starting point, x,y,q1,q2
    #x0, y0 = 0.1, -1.89
    x0, y0 = 1.4, 1.3
    #x0, y0 = 1.95, 0.1
    #x0, y0 = 0.2, 0.3
    #x0, y0 = 0.1, 1.89
    robot.joint1_angle, robot.joint2_angle = robot.inverse_kinematics(x0, y0)
    
    robot.set_L(L=np.array([[-1, 1, -1, -1], [1, -1, -1, -1]]))
    #robot.set_L(L=np.array([[1, 0, -1, 0], [0, 1, 0, -1]]))
    plot = False
    #plot = True

    for i in range(200):

        robot.trajectory.append(list(robot.get_end_effector_position()))
        robot.velocity.append(list(robot.get_end_effector_velocity()))

        if plot:
            # Plot the current state of the robot arm and the trajectory
            plt.clf()
            plt.xlim(robot.x_min, robot.x_max)
            plt.ylim(robot.y_min, robot.y_max)
            plt.plot([0, robot.link1_length * np.cos(robot.joint1_angle), robot.link1_length * np.cos(robot.joint1_angle) + robot.link2_length * np.cos(robot.joint1_angle + robot.joint2_angle)], [0, robot.link1_length * np.sin(robot.joint1_angle), robot.link1_length * np.sin(robot.joint1_angle) + robot.link2_length * np.sin(robot.joint1_angle + robot.joint2_angle)], '-o')
            plt.plot([pos[0] for pos in robot.trajectory], [pos[1] for pos in robot.trajectory], '-r')
            plt.grid()
            plt.gca().set_aspect("equal")
            plt.draw()
            plt.title('Teacher: Frame {}'.format(i+1))
            plt.pause(0.001)

        robot.update()

    trajectory = np.array(robot.trajectory)
    velocity = np.array(robot.velocity)

    # Create a figure with 4 subplots
    fig, axs = plt.subplots(4, 2)
    fig.suptitle('L = {}'.format(robot.L))

    idx = [i for i in range(len(robot.joint1_angle_track))]

    # Plot the lines on the subplots
    axs[0, 0].plot(idx, robot.joint1_angle_track)
    axs[0, 1].plot(idx, robot.joint2_angle_track)
    axs[1, 0].plot(idx, robot.joint1_velocity_track)
    axs[1, 1].plot(idx, robot.joint2_velocity_track)
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


    # Learning phase
    from sklearn.linear_model import LinearRegression  # Supports multi-output
    from sklearn.linear_model import Ridge  # Supports multi-output
    from sklearn.linear_model import ElasticNet  # Supports multi-output
    from sklearn.linear_model import LassoLars  # Supports multi-output
    from sklearn.linear_model import OrthogonalMatchingPursuit  # Supports multi-output
    from sklearn.kernel_ridge import KernelRidge  # Supports multi-output
    from sklearn.tree import DecisionTreeRegressor  # Supports multi-output
    from sklearn.ensemble import RandomForestRegressor  # Supports multi-output
    from sklearn.ensemble import ExtraTreesRegressor  # Supports multi-output
    from sklearn.ensemble import BaggingRegressor  # Supports multi-output

    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Lasso # Lasso regressor

    S = 4
    num = 10000 # experiments number, different combinations

    states = np.array(robot.state_track)
    torques = np.array(robot.torque_track)

    index_list = [x for x in range(len(states))]

    combinations_of_four = list(itertools.combinations(index_list, S))
    
    mses = []
    rmses = []
    r2s = []
    det_phi = []

    for item in random.sample(combinations_of_four, num):
        # Create instances of regression models
        # model = LinearRegression()
        model = Ridge(alpha=1.0)
        # model = make_pipeline(StandardScaler(with_mean=False), Lasso())
        # model = ElasticNet(alpha=1.0, l1_ratio=0.5)
        # model = make_pipeline(StandardScaler(with_mean=False), LassoLars())
        # model = make_pipeline(StandardScaler(with_mean=False), OrthogonalMatchingPursuit())
        # model = KernelRidge(alpha=1.0, kernel='linear')
        # model = DecisionTreeRegressor()

        ## Take too much time!!!
        # model = RandomForestRegressor(n_estimators=100)
        # model = ExtraTreesRegressor(n_estimators=100)
        # model = BaggingRegressor(n_estimators=10)

        # Train and evaluate the model
        results = robot.train_and_evaluate_learner(model, states, torques, train_indices=list(item))

        # normalize each row of train states to unit vector
        normalized_phi = states[list(item)] / np.linalg.norm(states[list(item)], axis=1, keepdims=True)

        det_phi.append(abs(np.linalg.det(normalized_phi.T)))
        
        mses.append(results['mean_squared_error'])
        rmses.append(results['root_mse'])
        r2s.append(results['r2_score'])

    plt.figure()
    plt.plot(det_phi, rmses, 'o')
    plt.title('det_phi vs RMSE')
    plt.xlabel('det_phi')
    plt.ylabel('RMSE')

    plt.show()

