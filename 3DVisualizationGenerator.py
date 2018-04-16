'''#######################################################
IMPORTS
'''  #######################################################
import logging
from torch.autograd import Variable
from BVHReader import *
from bvh import *
import argparse
from ModelTrainer import FeedForwardNet

parser = argparse.ArgumentParser(
    description='Runs a trained model on a bvh file to generate a blender visualization file.'
)
parser.add_argument('--model_filename',
                    type=str,
                    required=False,
                    default="model.pkl",
                    help='Name of file with network model.'
                    )
parser.add_argument('--bvh_filename',
                    type=str,
                    required=False,
                    default="../bvh_testdata/bvh_conversion/cmu_bvh/137/137_16.bvh",
                    help='Name of bvh file.'
                    )
parser.add_argument('--output_filename',
                    type=str,
                    required=False,
                    default="visualization_data",
                    help='Name of bvh file.'
                    )

args = parser.parse_args()

output_filename = args.output_filename + ".vdata"
bvh_filename = args.bvh_filename
model_filename = args.model_filename

# Load model and initialize dataset
model = torch.load(model_filename)
file_path_list = [bvh_filename]
m_dataset = BVHDatasetDeltas(file_path_list)
num_frames = len(m_dataset)
logging.info("{} frames loaded.".format(num_frames))

data_loader = data.DataLoader(dataset=m_dataset, batch_size=1, shuffle=False)

# Generate data
predicted_deltas = []
positions = []

# Predict Deltas
logging.info("Predicting Deltas")
for pose, delta in data_loader:
    pose = Variable(pose).float().cuda()
    pred = model(pose).data.cpu().numpy()[0]
    predicted_deltas.append(pred)

# Load positions
logging.info("Loading positions")
with open(bvh_filename) as f:
    bvh_data = Bvh(f.read())
for frame in bvh_data.frames[conf.NUM_INVALID_FRAMES+conf.NUM_FRAMES_LOOK_BEHIND:-conf.NUM_FRAMES_LOOK_AHEAD]:
    pos = [float(val) for val in frame[:3]]
    positions.append(pos)

logging.info("Verifying list lengths.")
try:
    assert (len(positions) == len(predicted_deltas))
except:
    logging.error(
        "Number of positions and predictions do not match! {}=/={}".format(len(positions), len(predicted_deltas))
    )
    exit()

logging.info("Writing to file")
with open(output_filename, "w+") as f:
    f.write(str(num_frames) + "\n")
    f.write(str(conf.NUM_FRAMES_LOOK_AHEAD) + "\n")
    f.write(bvh_filename + "\n")
    for i in range(len(positions)):
        to_write = list(positions[i]) + list(predicted_deltas[i])
        f.write(" ".join(map(str, to_write)))
        f.write("\n")
