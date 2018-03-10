'''#######################################################
IMPORTS
'''#######################################################
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from torch.autograd import Variable
from CustomBVHReader import BVHDataset
from bvh import *
import argparse

parser = argparse.ArgumentParser(
    description='Runs a trained model on a bvh file to generate a blender visualization file.')
parser.add_argument('--model_filename', metavar='m', type=str, required=False, default="model.pkl",
    help='Name of file with network model.')
parser.add_argument('--bvh_filename', metavar='b', type=str, required=False, default="bvh_testdata/bvh_conversion/cmu_bvh/08/08_01.bvh",
    help='Name of bvh file.')
parser.add_argument('--output_filename', metavar='o', type=str, required=False, default="visualization_data",
    help='Name of bvh file.')
args = parser.parse_args()

output_filename = args.output_filename
bvh_filename = args.bvh_filename
model_filename = args.model_filename

#Load model and initialize dataset
model = torch.load(model_filename)
file_path_list = [bvh_filename]
m_dataset = BVHDataset(file_path_list)
num_frames = len(m_dataset)
print("{} frames loaded.".format(num_frames))

data_loader = data.DataLoader(dataset=m_dataset, batch_size=1, shuffle=False)

# Generate data
pred_deltas= []
positions = []

# Predict Deltas
print("Predicting Deltas")
for pose, delta in data_loader:
    pose = Variable(pose).float()
    pred = model(pose).data.numpy()[0]
    pred_deltas.append( pred )

# Load positions
print("Loading positions")
with open(file_path_list[0]) as f:
    bvh_data = Bvh(f.read())
for frame in bvh_data.frames[1:-1]:
    pos = [float(val) for val in frame[:3]]
    positions.append( pos )

print("Verifying list lengths.")
assert(len(positions) == len(pred_deltas))

print("Writing to file")
with open(output_filename, "w+") as f:
    f.write(str(num_frames)+"\n")
    for i in range(len(positions)):
        for item in positions[i]:
            f.write(str(item) + " ")
        for item in pred_deltas[i]:
            f.write(str(item) + " ")
        f.write("\n")
print("Finished!")