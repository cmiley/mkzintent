'''#######################################################
IMPORTS
'''#######################################################
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from torch.autograd import Variable
from bvh import *
import os


'''#######################################################
CONSTANTS
'''#######################################################
# Number of frames at the begining of the file that are invalid.
NUM_INVALID_FRAMES = 1
NUM_FRAMES_LOOK_AHEAD = 1
NN_INPUT_SIZE = 93
NN_OUTPUT_SIZE = 3

class BVHDataset(data.Dataset):

    def __init__(self, file_paths):
        self.data = []

        for file_path in file_paths:
            with open(file_path) as f:
                bvh_data = Bvh(f.read())

            # One extra frame is discarded at the end
            num_samples = bvh_data.nframes - (NUM_INVALID_FRAMES + NUM_FRAMES_LOOK_AHEAD)

            for file_index in range(NUM_INVALID_FRAMES, num_samples + NUM_FRAMES_LOOK_AHEAD):
                bvh_data.frames[file_index] = map(float, bvh_data.frames[file_index])
                bvh_data.frames[file_index+NUM_FRAMES_LOOK_AHEAD] = map(float, bvh_data.frames[file_index+NUM_FRAMES_LOOK_AHEAD])

                pose = np.asarray(bvh_data.frames[file_index][NN_OUTPUT_SIZE:])
                initial = np.asarray(bvh_data.frames[file_index][:NN_OUTPUT_SIZE])
                final = np.asarray(bvh_data.frames[file_index+NUM_FRAMES_LOOK_AHEAD][:NN_OUTPUT_SIZE])
                delta = final - initial
                self.data.append( (initial, pose, delta) )

        self.length = len(self.data)
        
    def __getitem__(self, index):
        positiion, pose, delta = self.data[index]
        observed_output_data = torch.from_numpy(delta).float()
        input_data = torch.from_numpy(pose).float()

        return input_data,observed_output_data

    def __len__(self,):
        return self.length
    
def main():
    file_path_list = []
    white_list = []

    for root, dirs, files in os.walk("bvh_testdata/bvh_conversion/cmu_bvh"):
        for name in files:
            if name.endswith(".bvh") and check_for_nan(os.path.join(root, name)):
                white_list.append(os.path.join(root, name))

    with open('white_list', 'w+') as f:
        for name in white_list:
            f.write(name + '\n')

def check_for_nan(file_name):
    m_dataset = BVHDataset([file_name])

    data_loader = data.DataLoader(dataset=m_dataset, batch_size=32, shuffle=True)

    # Hidden layer size
    h1, h2, h3 = 200, 150, 50
    model = torch.nn.Sequential(
        nn.Linear(NN_INPUT_SIZE,h1),
        nn.ReLU(),
        nn.Linear(h1,h2),
        nn.ReLU(),
        nn.Linear(h2,h3),
        nn.ReLU(),
        nn.Linear(h3,NN_OUTPUT_SIZE)
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for i in range(10):
        for input_value, observed_output_value in data_loader:
            optimizer.zero_grad()
            input_value = Variable(input_value).float()
            observed_output_value = Variable(observed_output_value).float()

            pred = model(input_value)
            loss = criterion(pred, observed_output_value)
            loss_average = np.average(loss.data.numpy())

            loss.backward()
            optimizer.step()
            if np.isnan(loss_average):
                print('Caught failure')
                return False
    print('Success')
    return True


if __name__ == '__main__':
    main()