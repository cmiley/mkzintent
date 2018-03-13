'''#######################################################
IMPORTS
'''  #######################################################
import torch
import torch.utils.data as data
import numpy as np
from bvh import *

import os

'''#######################################################
CONSTANTS
'''  #######################################################
# Number of frames at the beginning of the file that are invalid.
NUM_INVALID_FRAMES = 5
NUM_FRAMES_LOOK_AHEAD = 60
NN_INPUT_SIZE = 93
NN_OUTPUT_SIZE = 3


class BVHDataset(data.Dataset):

    def __init__(self, file_paths):
        self.data = []

        for file_path in file_paths:
            with open(file_path) as f:
                bvh_data = Bvh(f.read())

            for file_index in range(NUM_INVALID_FRAMES, bvh_data.nframes - NUM_FRAMES_LOOK_AHEAD):
                bvh_data.frames[file_index] = map(float, bvh_data.frames[file_index])
                bvh_data.frames[file_index + NUM_FRAMES_LOOK_AHEAD] = map(float, bvh_data.frames[
                    file_index + NUM_FRAMES_LOOK_AHEAD])

                pose = np.asarray(bvh_data.frames[file_index][NN_OUTPUT_SIZE:])
                initial = np.asarray(bvh_data.frames[file_index][:NN_OUTPUT_SIZE])
                final = np.asarray(bvh_data.frames[file_index + NUM_FRAMES_LOOK_AHEAD][:NN_OUTPUT_SIZE])
                delta = final - initial
                self.data.append((initial, pose, delta))

        self.length = len(self.data)

    def __getitem__(self, index):
        position, pose, delta = self.data[index]
        observed_output_data = torch.from_numpy(delta).float()
        input_data = torch.from_numpy(pose).float()

        return input_data, observed_output_data

    def __len__(self, ):
        return self.length


# TODO Argparse whitelist.txt
def load_whitelist():
    paths = []
    with open("white_list.txt", 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        paths.append(line)
    return paths
