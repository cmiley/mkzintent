'''
#######################################################
IMPORTS
#######################################################
'''
import torch
import torch.utils.data as data
import numpy as np
from bvh import *
import MKZIntentConf as conf


class BVHDataset(data.Dataset):

    def __init__(self, file_paths):
        self.data = []

        for file_path in file_paths:
            with open(file_path) as f:
                bvh_data = Bvh(f.read())

            bvh_data.frames = convert_frames_to_float(bvh_data.frames)

            for file_index in range(conf.NUM_INVALID_FRAMES, bvh_data.nframes - conf.NUM_FRAMES_LOOK_AHEAD):

                pose = np.asarray(bvh_data.frames[file_index][conf.NN_OUTPUT_SIZE:])
                initial = np.asarray(bvh_data.frames[file_index][:conf.NN_OUTPUT_SIZE])
                final = np.asarray(bvh_data.frames[file_index + conf.NUM_FRAMES_LOOK_AHEAD][:conf.NN_OUTPUT_SIZE])
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


class BVHRNNDataset(data.Dataset):

    def __init__(self, file_paths, sequence_size):
        self.data = []
        self.seq_map = [(0, 0)]
        self.sequence_size = sequence_size

        frame_count = 0
        seq_count = 0
        for file_path in file_paths:
            with open(file_path) as f:
                bvh_data = Bvh(f.read())

            bvh_data.frames = convert_frames_to_float(bvh_data.frames)

            for file_index in range(conf.NUM_INVALID_FRAMES, bvh_data.nframes - conf.NUM_FRAMES_LOOK_AHEAD):
                pose = np.asarray(bvh_data.frames[file_index][conf.NN_OUTPUT_SIZE:])
                initial = np.asarray(bvh_data.frames[file_index][:conf.NN_OUTPUT_SIZE])
                final = np.asarray(bvh_data.frames[file_index + conf.NUM_FRAMES_LOOK_AHEAD][:conf.NN_OUTPUT_SIZE])
                delta = final - initial
                self.data.append((initial, pose, delta))

                frame_count += 1
                seq_count += 1

            seq_count -= sequence_size - 1
            self.seq_map.append((seq_count, frame_count))

        self.length = self.seq_map[-1][0]

    def get_index(self, input_index):
        i = 0
        while i < len(self.seq_map) and self.seq_map[i][0] <= input_index:
            i += 1
        i -= 1
        map_tuple = self.seq_map[i]
        offset = input_index - map_tuple[0]
        return map_tuple[1] + offset

    def __getitem__(self, index):

        return_sequence = []
        # Loop over some sequence length to get multiple pose/output pairs
        start_index = self.get_index(index)
        for i in range(start_index, start_index + self.sequence_size):
            position, pose, delta = self.data[i]
            observed_output_data = torch.from_numpy(delta).float()
            input_data = torch.from_numpy(pose).float()

            return_sequence.append([input_data, observed_output_data])

        # We want a sequence of input and output data, not just one.
        return return_sequence

    def __len__(self, ):
        return self.length


class BVHDatasetDeltas(data.Dataset):

    def __init__(self, file_paths):
        self.data = []

        for file_path in file_paths:
            with open(file_path) as f:
                bvh_data = Bvh(f.read())

            bvh_data.frames = convert_frames_to_float(bvh_data.frames)

            for file_index in range(conf.NUM_INVALID_FRAMES+conf.NUM_FRAMES_LOOK_BEHIND, bvh_data.nframes - conf.NUM_FRAMES_LOOK_AHEAD):

                pose = np.asarray(bvh_data.frames[file_index][conf.NN_OUTPUT_SIZE:])
                prior = np.asarray(bvh_data.frames[file_index - conf.NUM_FRAMES_LOOK_BEHIND][:conf.NN_OUTPUT_SIZE])
                current = np.asarray(bvh_data.frames[file_index][:conf.NN_OUTPUT_SIZE])
                final = np.asarray(bvh_data.frames[file_index + conf.NUM_FRAMES_LOOK_AHEAD][:conf.NN_OUTPUT_SIZE])
                delta_future = final - current
                delta_prior = (current - prior)/(bvh_data.frame_time*conf.NUM_FRAMES_LOOK_BEHIND)
                pose_with_delta = np.hstack((pose, delta_prior))
                self.data.append((current, pose_with_delta, delta_future))

        self.length = len(self.data)

    def __getitem__(self, index):
        position, pose, delta = self.data[index]
        observed_output_data = torch.from_numpy(delta).float()
        input_data = torch.from_numpy(pose).float()

        return input_data, observed_output_data

    def __len__(self, ):
        return self.length


# Converts an array of frames stored as strings to an array of frames stored as floats.
# Returns the modified array.
def convert_frames_to_float(frames):
    for frame_num in range(len(frames)):
        frames[frame_num] = list(map(float, frames[frame_num]))
    return frames


# TODO Argparse whitelist.txt
def load_whitelist():
    paths = []
    with open(conf.WHITE_LIST_FILE, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        paths.append(line)
    return paths
