import os
import numpy as np
import random
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm

def generate_phi(num, min_velocity_arg, max_velocity_arg):
    # Initialize an empty 4x3 matrix
    matrix = np.zeros((4,num))

    for j in range(num): # for each column
        while True: # keep trying until we find valid a and b
            a = random.uniform(0, 2) # random float between 0 and 2
            b = random.uniform(0, 2) # random float between 0 and 2
            # Check the condition
            if a**2 + b**2 <= 4:
                matrix[0,j] = a
                matrix[1,j] = b
                break

    for i in range(2, 4): # for the third and fourth row
        for j in range(num): # for each column
            matrix[i,j] = random.uniform(min_velocity_arg, max_velocity_arg) # random float between 0 and 0.5

    matrix = np.vstack((matrix, np.ones((1, 5))))
    return matrix

def generate_force(min_force_arg, max_force_arg):
    force = np.random.uniform(min_force_arg, max_force_arg, size=(2, 5))
    return force

def generate_data(phi_num, skill_num, min_velocity_arg, max_velocity_arg, min_force_arg, max_force_arg, dataset_num):
    phi_list = []
    force_random_list = []
    
    if skill_num == 1:
        real_L = np.array([[1, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0]])
    elif skill_num == 2:
        real_L = np.array([[-1, 0, -2, 0, 0.8],
                        [0, -1, 0, -2, 1.2]])
    else:
        print("Input correct skill num.")

    for num in tqdm(range(phi_num)):
        phi = generate_phi(5, min_velocity_arg, max_velocity_arg)
        phi_list.append(phi)

        force_random = generate_force(min_force_arg, max_force_arg)
        force_random_list.append(force_random)

    # set path
    current_path_dir = os.path.dirname(os.path.abspath(__file__))
    current_path = os.path.join(current_path_dir, f'artificial_dataset\{dataset_num}')
    os.makedirs(current_path, exist_ok=True)

    file_path_phi = os.path.join(current_path, 'dataset_phi.npy')
    np.save(file_path_phi, phi_list)

    file_path_force_random = os.path.join(current_path, 'dataset_force_random.npy')
    np.save(file_path_force_random, force_random_list) #(1000, 2, 5)

    # save ReadMe file
    filename = os.path.join(current_path, 'ReadMe.txt')
    with open(filename, 'w') as file:
        file.write(f'{dataset_num} is learning skill {skill_num}.\n\n')

        file.write(f'data number = {data_num}\n')
        file.write(f'real L = \n')
        file.write(f'{real_L[0]}\n')
        file.write(f'{real_L[1]}\n')
    
        file.write(f'min velocity    = {min_velocity_arg}\n')
        file.write(f'max velocity    = {max_velocity_arg}\n')

        file.write(f'min force       = {min_force_arg}\n')
        file.write(f'max force       = {max_force_arg}\n')
    print(f"Dataset file saved to {current_path}")

if __name__ == '__main__':
    data_num = 10000
    skill_num = 1

    min_velocity_value = -0.3
    max_velocity_value = 0.3

    min_force_value = -1.0
    max_force_value = 1.0

    dataset_name = 'dataset2'

    generate_data(data_num, skill_num, min_velocity_value, max_velocity_value, min_force_value, max_force_value, dataset_name)

