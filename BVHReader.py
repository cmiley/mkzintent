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
import datetime


'''#######################################################
CONSTANTS
'''#######################################################
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

            # One extra frame is discarded at the end
            num_samples = bvh_data.nframes - (NUM_INVALID_FRAMES + NUM_FRAMES_LOOK_AHEAD)

            for file_index in range(NUM_INVALID_FRAMES, bvh_data.nframes - NUM_FRAMES_LOOK_AHEAD):
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
    lr = (0.01 ** ((epoch+3) //3))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def load_whitelist():
    paths = []
    with open("white_list", 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        paths.append(line)
    return paths

def evaluate_model_on_dataset(model, criterion, dataset):

    loader = data.DataLoader(dataset=dataset, batch_size=2**12, shuffle=True)

    loss_sum = 0.0
    num_samples = 0.0
    for input_value, observed_output_value in loader:
        input_value = Variable(input_value).float()
        observed_output_value = Variable(observed_output_value).float()

        pred = model(input_value)
        loss = criterion(pred, observed_output_value)
        loss_sum += np.average(loss.data.numpy())
        num_samples += 1.0
    return loss_sum/num_samples

def main():

    file_path_list = load_whitelist()[:50]
    
    # for i in range(1):
    #     for j in range(8,9):
    #         file_path_list.append(
    #             "bvh_testdata/bvh_conversion/cmu_bvh/{}/{}_{}.bvh".format(str(j).zfill(2), str(j).zfill(2), str(i+1).zfill(2))
    #         )

    # for root, dirs, files in os.walk("bvh_testdata/bvh_conversion/cmu_bvh"):
    #     for name in files:
    #         if name.endswith(".bvh"):
    #             file_path_list.append(os.path.join(root, name))

    print("Start time: {}".format(datetime.datetime.now()))
    print(str(len(file_path_list)) + " filenames loaded.")


    split_index = int(len(file_path_list)*0.9)
    train_filelist = file_path_list[:split_index]
    test_filelist = file_path_list[split_index:]

    train_dataset = BVHDataset(train_filelist)
    test_dataset = BVHDataset(test_filelist)
    print("Data has been indexed.")

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=2**12, shuffle=True, num_workers=4)

    train_loss_history = []
    test_loss_history = []

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

    for i in range(10):
        print("Start of epoch {}".format(i+1))
        start = time.time()
        adjust_learning_rate(optimizer, i+1)
        for input_value, observed_output_value in train_loader:

            optimizer.zero_grad()
            
            input_value = Variable(input_value).float()
            observed_output_value = Variable(observed_output_value).float()
            
            pred = model(input_value)
            loss = criterion(pred, observed_output_value)
            loss.backward()
            optimizer.step()

        train_loss = evaluate_model_on_dataset(model, criterion, train_dataset)
        test_loss = evaluate_model_on_dataset(model, criterion, test_dataset)
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)

        print("Training loss: {}\tTesting loss:{}".format(train_loss, test_loss))
        print("Time for iteration {}".format(time.time() - start))

    print("Completed training...")
    print("End time: {}".format(datetime.datetime.now()))
    model_filename = "model.pkl"
    print("Saving model to {}".format(model_filename))
    torch.save(model, model_filename)

    train_loss = evaluate_model_on_dataset(model, criterion, train_dataset)
    test_loss = evaluate_model_on_dataset(model, criterion, test_dataset)
    print("Training loss: {}\tTesting loss:{}".format(train_loss, test_loss))

    plt.figure(1)
    plt.title("Loss")
    plt.plot(range(len(train_loss_history)), train_loss_history, label = "Training Loss")
    plt.plot(range(len(test_loss_history)), test_loss_history, label = "Testing Loss")
    plt.yscale('log')
    plt.legend()

    plt.show()

if __name__ == '__main__':
    main()