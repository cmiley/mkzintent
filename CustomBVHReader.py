'''#######################################################
IMPORTS
'''#######################################################
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from bvh import *
import time


'''#######################################################
CONSTANTS
'''#######################################################
# Number of frames at the begining of the file that are invalid.
NUM_INVALID_FRAMES = 1
NN_INPUT_SIZE = 93
NN_OUTPUT_SIZE = 3

class BVHDataset(data.Dataset):

    def __init__(self, file_paths):
        self.length = 0
        self.file_index_bounds = []

        for file_path in file_paths:
            with open(file_path) as f:
                bvh_data = Bvh(f.read())

            # One extra frame is discarded at the end
            num_samples = bvh_data.nframes - (NUM_INVALID_FRAMES + 1)
            start_index = self.length
            end_index = start_index + num_samples - 1
            self.file_index_bounds.append( (file_path, start_index, end_index) )
            # read length of bvh_data
            self.length += num_samples
        

    def __getitem__(self, index):
        # determine triple
        item_triple = None

        for item in self.file_index_bounds:
            if index <= item[2]:
                item_triple = item
                break

        # open file and pull index
        fname = item_triple[0]
        with open(fname) as f:
            bvh_data = Bvh(f.read())

        file_index = index - item_triple[1] + NUM_INVALID_FRAMES

        bvh_data.frames[file_index] = map(float, bvh_data.frames[file_index])
        bvh_data.frames[file_index+1] = map(float, bvh_data.frames[file_index+1])

        found_frame = np.asarray(bvh_data.frames[file_index][NN_OUTPUT_SIZE:])
        initial = np.asarray(bvh_data.frames[file_index][:NN_OUTPUT_SIZE])
        final = np.asarray(bvh_data.frames[file_index+1][:NN_OUTPUT_SIZE])

        delta = final - initial
        observed_output_data = torch.from_numpy(delta).float()
        input_data = torch.from_numpy(found_frame).float()

        return input_data,observed_output_data

    def __len__(self,):
        return self.length
    
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = (0.01 ** ((epoch+3) //3))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    file_path_list = []
    for i in range(11):
        file_path_list.append("bvh_testdata/bvh_conversion/cmu_bvh/08/08_" + str(i+1).zfill(2) + ".bvh")

    m_dataset = BVHDataset(file_path_list)

    data_loader = data.DataLoader(dataset=m_dataset, batch_size=200, shuffle=True)
    # for input_value, output_value in data_loader:
    #     print(input_value, output_value)
    #     break

    loss_history = []

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
        start = time.time()
        adjust_learning_rate(optimizer, i+1)
        for input_value, observed_output_value in data_loader:
            optimizer.zero_grad()
            input_value = Variable(input_value).float()
            observed_output_value = Variable(observed_output_value).float()

            pred = model(input_value)
            loss = criterion(pred, observed_output_value)
            loss_history.append(loss.data[0])

            loss.backward()
            optimizer.step()
        print("loss:", loss.data[0])
        print("Time for iteration {}".format(time.time() - start))

    plt.figure(1)
    plt.title("Loss")
    plt.plot(range(len(loss_history)), loss_history)
    plt.yscale('log')

    plt.show()

if __name__ == '__main__':
    main()