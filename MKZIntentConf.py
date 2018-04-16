# Number of frames at the beginning of the file that are invalid.
NUM_INVALID_FRAMES = 5
NUM_FRAMES_LOOK_AHEAD = 80
SAMPLE_LOOK_AHEAD = 1
NN_INPUT_SIZE = 93
NN_OUTPUT_SIZE = 3
NUM_FRAMES_LOOK_BEHIND = 80

RNN_SEQUENCE_SIZE = 10
RNN_RECURRENT_SIZE = 30

LOG_DIR = "logs"
WHITE_LIST_LOG = LOG_DIR + "/whitelist_log"
DATA_DIR = "bvh_testdata"

WHITE_LIST_FILE = "white_list"

BATCH_SIZE = 2**12
NUM_EPOCHS = 50
