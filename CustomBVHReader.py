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

            for file_index in range(NUM_INVALID_FRAMES, num_samples - NUM_FRAMES_LOOK_AHEAD):
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
    
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = (0.005 ** ((epoch+3) //3))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    file_path_list = []
    for i in range(11):
        for j in range(5,10):
            file_path_list.append(
                "bvh_testdata/bvh_conversion/cmu_bvh/{}/{}_{}.bvh".format(str(j).zfill(2), str(j).zfill(2), str(i+1).zfill(2))
            )

    # for root, dirs, files in os.walk("bvh_testdata/bvh_conversion/cmu_bvh"):
    #     for name in files:
    #         if name.endswith(".bvh"):
    #             file_path_list.append(os.path.join(root, name))

    print("Filenames loaded.")
    m_dataset = BVHDataset(file_path_list)
    print("Data has been indexed.")

    data_loader = data.DataLoader(dataset=m_dataset, batch_size=80, shuffle=True, num_workers=6)

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

    print("Starting training.")
    for i in range(3):
        print("Start of epoch {}".format(i+1))
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

    model_filename = "model.pkl"
    print("Saving model to {}".format(model_filename))
    torch.save(model, model_filename)

    plt.figure(1)
    plt.title("Loss")
    plt.plot(range(len(loss_history)), loss_history)
    plt.yscale('log')

    plt.show()

if __name__ == '__main__':
    main()