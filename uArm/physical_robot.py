import os
import sys
import json
import math
import pygame
import pyttsx3
import keyboard
import numpy as np

RECORDS = {
            'datas': [],
            'count': 0
        }

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
    
# Screen specs
screen_width_cm = 51.8
screen_height_cm = 32.5
screen_offsets = np.array([-0.5, -2.1])

#
LEARNING_CNT = 8

# Setup the uArm
sys.path.append('..')
from uarm.wrapper import SwiftAPI

speaker = pyttsx3.init()
speaker.setProperty('rate', 200) 

swift = SwiftAPI(port="COM8", callback_thread_pool_size=1)

# Initialize Pygame
pygame.init()

# Set up the display for full screen
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
pygame.display.set_caption("Working Space")

# Get screen dimensions
screen_width, screen_height = screen.get_size()
pixels_per_cm_x = screen_width / screen_width_cm
pixels_per_cm_y = screen_height / screen_height_cm
# print(f"Pixels per cm: {pixels_per_cm_x:.2f} (x-axis), {pixels_per_cm_y:.2f} (y-axis)")

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 100, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (50, 50, 50)
LIGHT_GREEN = (144, 238, 144)  # Light green color
PINK = (255, 182, 193)         # Pink color
BLUE = (0, 0, 255)             # Blue color

DASHED_RADIUS = 5



# Fill the screen with white
screen.fill(WHITE)

def cm_to_pixels(cm_x, cm_y):
    px_x = cm_x * pixels_per_cm_x
    px_y = screen_height - (cm_y * pixels_per_cm_y)
    return px_x, px_y

# Draw an arrow function
def draw_arrow(screen, color, start_cm, end_cm, width=5):
    start_px = cm_to_pixels(*start_cm)
    end_px = cm_to_pixels(*end_cm)
    pygame.draw.line(screen, color, start_px, end_px, width)
    arrow_head = pygame.Vector2(end_px) - pygame.Vector2(start_px)
    arrow_head.scale_to_length(20)
    arrow_side1 = arrow_head.rotate(135)
    arrow_side2 = arrow_head.rotate(-135)
    pygame.draw.line(screen, color, end_px, end_px + arrow_side1, width)
    pygame.draw.line(screen, color, end_px, end_px + arrow_side2, width)

# Draw a circle function
def draw_circle(screen, color, center_cm, radius_cm):
    center_px = cm_to_pixels(*center_cm)
    radius_px = cm_to_pixels(radius_cm, 0)[0]  # Only take the x component for radius
    pygame.draw.circle(screen, color, center_px, radius_px)

# Draw a dashed circle function
def draw_dashed_circle(screen, color, center_cm, radius_cm, dash_length_cm=0.5):
    center_px = cm_to_pixels(*center_cm)
    radius_px = cm_to_pixels(radius_cm, 0)[0]  # Only take the x component for radius
    dash_length_px = cm_to_pixels(dash_length_cm, 0)[0]  # Only take the x component for dash length
    num_dashes = int(2 * math.pi * radius_px / dash_length_px)
    for i in range(num_dashes):
        start_angle = i * (2 * math.pi / num_dashes)
        end_angle = start_angle + (2 * math.pi / num_dashes) / 2
        start_pos = (center_px[0] + radius_px * math.cos(start_angle), center_px[1] + radius_px * math.sin(start_angle))
        end_pos = (center_px[0] + radius_px * math.cos(end_angle), center_px[1] + radius_px * math.sin(end_angle))
        pygame.draw.line(screen, color, start_pos, end_pos, 1)

# Draw rounded rectangle function
def draw_rounded_rect(surface, color, rect_cm, radius_cm):
    x_cm, y_cm, w_cm, h_cm = rect_cm
    x_px, y_px = cm_to_pixels(x_cm, y_cm)
    w_px, h_px = cm_to_pixels(w_cm, h_cm)
    radius_px = cm_to_pixels(radius_cm, 0)[0]
    pygame.draw.rect(surface, color, (x_px + radius_px, y_px, w_px - 2 * radius_px, h_px))
    pygame.draw.rect(surface, color, (x_px, y_px + radius_px, w_px, h_px - 2 * radius_px))
    pygame.draw.circle(surface, color, (x_px + radius_px, y_px + radius_px), radius_px)
    pygame.draw.circle(surface, color, (x_px + w_px - radius_px, y_px + radius_px), radius_px)
    pygame.draw.circle(surface, color, (x_px + radius_px, y_px + h_px - radius_px), radius_px)
    pygame.draw.circle(surface, color, (x_px + w_px - radius_px, y_px + h_px - radius_px), radius_px)

def show_finished_phase():
    # Display the pop-up window with shadow
    shadow_offset = 10
    popup_rect = pygame.Rect(screen_width // 2 - 200, screen_height // 2 - 100, 400, 200)
    shadow_rect = popup_rect.move(shadow_offset, shadow_offset)
    draw_rounded_rect(screen, DARK_GRAY, shadow_rect, 20)
    draw_rounded_rect(screen, WHITE, popup_rect, 20)
    pygame.draw.rect(screen, BLACK, popup_rect, 2, border_radius=20)

    # Display the text
    font = pygame.font.Font(None, 50)
    text = font.render("The phase is finished!", True, BLACK)
    text_rect = text.get_rect(center=(screen_width // 2, screen_height // 2 - 40))
    screen.blit(text, text_rect)

    # Create a button with rounded corners
    button_font = pygame.font.Font(None, 40)
    button_text = button_font.render("Close", True, WHITE)
    button_rect = pygame.Rect(screen_width // 2 - 75, screen_height // 2 + 30, 150, 50)
    draw_rounded_rect(screen, GREEN, button_rect, 10)
    pygame.draw.rect(screen, BLACK, button_rect, 2, border_radius=10)
    button_text_rect = button_text.get_rect(center=button_rect.center)
    screen.blit(button_text, button_text_rect)

    pygame.display.flip()

    # Wait for button click
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if button_rect.collidepoint(event.pos):
                    pygame.quit()
                    sys.exit()

def learn(states, actions, demo=5, lam=1e-6):
    lam = lam
    I = np.identity(demo)

    states[:2] = states[:2] / 10
    actions = actions / 10

    learned_thetea = (np.linalg.inv(states @ states.T + lam * I) @ states @ (actions.T)).T
    learned_thetea[:, -1] = learned_thetea[:, -1] * 10
    return learned_thetea


def store_data_tojson(phi, real_force, user_force, real_theta, learned_theta, force_guidance, el2):
        global RECORDS

        current_dir = os.path.dirname(os.path.abspath(__file__))
        # target_dir = os.path.join(os.path.join(current_dir, f'formal_exp_results'), f"formal_exp_id_user_000")
        # filename = os.path.join(target_dir, f"formal_exp_id_user_000.json")

        target_dir = os.path.join(os.path.join(current_dir, f'formal_exp_results'), f"formal_exp_id_user_03")
        filename = os.path.join(target_dir, f"formal_exp_id_user_03.json")

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        if os.path.exists(filename):
            with open(filename, 'r') as f:
                RECORDS = json.load(f)


        RECORDS['datas'].append({
            'phi': phi.tolist(),
            'real_force': real_force.tolist(),
            'user_force': user_force.tolist(),
            'real_theta': real_theta.tolist(),
            'learned_theta': learned_theta.tolist(),
            'force_score': force_guidance,
            'el2': el2
        })
        RECORDS['count'] += 1
        
        with open(filename, 'w') as f:
            json.dump(RECORDS, f, cls=NumpyEncoder)

def experiments(elements, target_position, score, real_cal_theta):
    given_actions = []
    given_states = []
    for idx in elements:
        draw_circle(screen, RED, target_position+screen_offsets, 0.5)
        pygame.display.flip()
        swift.reset(x=elements[idx]['circle'][0]*10, y=0, z=elements[idx]['circle'][1]*10, speed=20000)
        circle = elements[idx]['circle']
        draw_circle(screen, BLACK, circle+screen_offsets, 0.1)
        draw_dashed_circle(screen, GRAY, circle+screen_offsets, DASHED_RADIUS)
        pygame.display.flip()

        speaker.say(f'Trial {idx}. Start!')
        speaker.runAndWait()
        keyboard.wait('m')

        swift.set_servo_detach(servo_id=1)
        swift.set_servo_detach(servo_id=2)

        while True:
            positions = swift.angles_to_coordinate(angles=swift.get_servo_angle())

            if np.linalg.norm([positions[0]-elements[idx]['circle'][0]*10, positions[2]-elements[idx]['circle'][1]*10]) > 50:
                    swift.set_servo_attach()

                    speaker.say("Oops, out of the limits. Press 'R' to move back.")
                    speaker.runAndWait()
                    keyboard.wait('r')

                    swift.reset(x=elements[idx]['circle'][0]*10, y=0, z=elements[idx]['circle'][1]*10)

                    speaker.say("press 'M' to start moving the robot again.")
                    speaker.runAndWait()
                    keyboard.wait('m')

                    swift.set_servo_detach(servo_id=1)
                    swift.set_servo_detach(servo_id=2)

            # Non-blocking check for Enter key press
            if keyboard.is_pressed('s'):
                    ux = positions[0]/10-elements[idx]['circle'][0]
                    uy = positions[2]/10-elements[idx]['circle'][1]
                    given_states.append([elements[idx]['circle'][0], elements[idx]['circle'][1], 1])
                    given_actions.append([ux, uy])
                    print(f"Robot stopped at position: [{positions[0]/10}, {positions[2]/10}]")
                    print(f"Give actions are: {np.array([positions[0]/10-elements[idx]['circle'][0], positions[2]/10-elements[idx]['circle'][1], 1])}")
                    print(f"Real actions are: {np.array([elements[idx]['arrow'][0][0]-elements[idx]['arrow'][1][0], elements[idx]['arrow'][0][1]-elements[idx]['arrow'][1][1], 1])}")

                    swift.set_servo_attach()

                    r_actions = elements[idx]['arrow']
                    draw_arrow(screen, LIGHT_GREEN, r_actions[0, :]+screen_offsets, np.array([r_actions[0, 0]+ux, r_actions[0, 1]+uy])+screen_offsets)
                    pygame.display.flip()

                    if score:
                        draw_arrow(screen, DARK_GREEN, r_actions[0, :]+screen_offsets, r_actions[1, :]+screen_offsets)
                        pygame.display.flip()

                    break

    given_actions = np.array(given_actions).T
    given_states = np.array(given_states).T
    print('actions', given_actions)
    
    theta = learn(given_states, given_actions, demo=3)
    print("given_states", given_states)

    el2 = np.linalg.norm(real_cal_theta - theta)
    print('el2', el2)
    print('theta', theta)

    speaker.say('Finish!')
    speaker.runAndWait()
    #######################################################################################
    # Student Demonstration
    ######################################################################################
    swift.reset(x=130, y=0, z=75) # reset robot end-effector to a starting point
    circle = np.array([13, 7.5])
    draw_circle(screen, PINK, circle+screen_offsets, 0.1)
    pygame.display.flip()
    speaker.say('Let us see the learned movement!')
    speaker.runAndWait()
    x, _, z = swift.get_position(timeout=100000) # get the current x, z position. y is ignored because we fixed the operation coordinate
    x /= 10
    z /= 10
    vel = 100 # fixed velocity in mm/second 

    steps = 0 # update step No.
    trajectory = [] # buffer for storing the trajectory

    while steps < 50: # or while position_difference < aThreshold:

        # using THETA without velocity term
        delta_x, delta_z = theta @ np.array([x, z, 1]).T

        ## Update equations
        x += delta_x
        z += delta_z
        
        swift.set_position(x=x*10, y=0, z=z*10, speed=vel*60) # speed: degree/min

        ## Get real positions
        x, _, z = swift.get_position(timeout=100000)
        x /= 10
        z /= 10 
        trajectory.append([x, z])

        steps += 1

    print('Target positions: ', target_position, '; Current positions: ', [x, z])

    return given_actions, given_states, theta, el2, trajectory

# Initialize index for elements
elements = {
    1: {'circle': np.array([13, 7.5]), 'arrow': np.array([[13, 7.5], [0, 0]])},
    2: {'circle': np.array([27, 10.5]), 'arrow': np.array([[27, 10.5], [0, 0]])},
    3: {'circle': np.array([20, 16.5]), 'arrow': np.array([[20, 16.5], [0, 0]])}
}

pre_elements = {
    1: {'circle': np.array([32, 5]), 'arrow': np.array([[32, 5], [0, 0]])},
    2: {'circle': np.array([26, 17]), 'arrow': np.array([[26, 17], [0, 0]])},
    3: {'circle': np.array([21, 7]), 'arrow': np.array([[21, 7], [0, 0]])}
}

# Draw the target point
target_position = np.array([21, 11])
pre_target_position = np.array([23, 12])

# Real controller
real_theta = np.array([[-0.5, 0, 0.5*target_position[0]],
                       [0, -0.5, 0.5*target_position[1]]])


pre_real_theta = np.array([[-0.5, 0, 0.5*pre_target_position[0]],
                       [0, -0.5, 0.5*pre_target_position[1]]])




states_np = np.array([
     [elements[1]['circle'][0], elements[2]['circle'][0], elements[3]['circle'][0]],
     [elements[1]['circle'][1], elements[2]['circle'][1], elements[3]['circle'][1]],
     [1, 1, 1]
])

pre_states_np = np.array([
     [pre_elements[1]['circle'][0], pre_elements[2]['circle'][0], pre_elements[3]['circle'][0]],
     [pre_elements[1]['circle'][1], pre_elements[2]['circle'][1], pre_elements[3]['circle'][1]],
     [1, 1, 1]
])

real_actions = real_theta @ states_np
for idx in elements:
    elements[idx]['arrow'][1, 0] = elements[idx]['arrow'][0, 0] + real_actions[0, idx-1]
    elements[idx]['arrow'][1, 1] = elements[idx]['arrow'][0, 1] + real_actions[1, idx-1]

pre_real_actions = pre_real_theta @ pre_states_np
for idx in pre_elements:
    pre_elements[idx]['arrow'][1, 0] = pre_elements[idx]['arrow'][0, 0] + pre_real_actions[0, idx-1]
    pre_elements[idx]['arrow'][1, 1] = pre_elements[idx]['arrow'][0, 1] + pre_real_actions[1, idx-1]

speaker.say('Now, let us start the formal experiments.')
# speaker.say("During each trial, press 'M' to start moving the robot. Press S whenever you think you have reached a correct position")
speaker.runAndWait()

speaker.say('Exam stage.')
speaker.runAndWait()

actions = {}
states = {}
thetas = {}
el2s = {}
trajectories = {}

GUIDANCE_FLAG = False
given_actions, given_states, theta, el2, trajectory = experiments(pre_elements, pre_target_position, GUIDANCE_FLAG, pre_real_theta)
screen.fill(WHITE)

store_data_tojson(given_states, real_actions, given_actions, pre_real_theta, theta, GUIDANCE_FLAG, el2)


# actions['exam1'] = actions
# states['exam1'] = states
# thetas['exam1'] = theta
# el2s['exam1'] = el2
# trajectories['exam1'] = trajectory

GUIDANCE_FLAG = True
for i in range(1, LEARNING_CNT+1):
    speaker.say(f'Learning stage {i}.')
    speaker.runAndWait()
    given_actions, given_states, theta, el2, trajectory = experiments(elements, target_position, GUIDANCE_FLAG, real_theta)
    screen.fill(WHITE)

    store_data_tojson(given_states, real_actions, given_actions, real_theta, theta, GUIDANCE_FLAG, el2)
    

    # actions[f'learning{i}'] = actions
    # states[f'learning{i}'] = states
    # thetas[f'learning{i}'] = theta
    # el2s[f'learning{i}'] = el2
    # trajectories[f'learning{i}'] = trajectory

speaker.say('Exam stage.')
speaker.runAndWait()
GUIDANCE_FLAG = False
given_actions, given_states, theta, el2, trajectory = experiments(pre_elements, pre_target_position, GUIDANCE_FLAG, real_theta)

store_data_tojson(given_states, real_actions, given_actions, pre_real_theta, theta, GUIDANCE_FLAG, el2)


# actions['exam2'] = actions
# states['exam2'] = states
# thetas['exam2'] = theta
# el2s['exam2'] = el2
# trajectories['exam2'] = trajectory
# # Convert trajectory to separate x and y lists
# x_points = [point[0] for point in trajectory]
# y_points = [point[1] for point in trajectory]
