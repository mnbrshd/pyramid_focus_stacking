import numpy as np

kernel_sharp = np.array([[0, -1, 0], 
                         [-1, 5, -1],
                         [0, -1, 0]])

    
kernel_gauss = np.array([[1, 2, 1], 
                         [2, 4, 2],
                         [1, 2, 1]]) / 16

sharpen_kernel = np.array([[-1,-1,-1],
                           [-1,9,-1],
                           [-1,-1,-1]])

num_pyramids = 5

p0_list = [1, 1.25, 1.5, 1.75, 2]
