import sys

import os
import torch.nn as nn

import time
import datetime
import logging

from BVHReader import *
from ModelEvaluationTools import *

LOG_DIR = "logs/"

LOSS_LOG_LEVEL = logging.INFO
TIMING_LOG_LEVEL = logging.INFO
IO_LOG_LEVEL = logging.NOTSET

LOSS_TITLE = "Loss Graph"
ITERATION_TITLE = "Iteration Graph"


class RNN(nn.Module):
    def __init__(self, input_size, recurrent_size, output_size):
        super(RNN, self).__init__()

        self.recurrent_size = recurrent_size

        self.recurrent_layer = nn.Linear(input_size + recurrent_size, recurrent_size)

        # Hidden layer size
        h1, h2, h3 = 200, 150, 50
        self.feed_forward_layers = torch.nn.Sequential(
            nn.Linear(input_size + recurrent_size, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Linear(h3, output_size)
        )

    def forward(self, input_value, recurrent):
        combined = torch.cat((input_value, recurrent), 1)

        recurrent = self.recurrent_layer(combined)
        output = self.feed_forward_layers(combined)
        return output, recurrent

    def init_recurrent(self, n):
        return Variable(torch.zeros(n, self.recurrent_size))


def save_test_train_split(path, train_file_list, test_file_list):
    split_info = {
        "train_file_list": train_file_list,
        "test_file_list": test_file_list
    }
    import json
    with open(os.path.join(path, "test_train_split_info"), "w+") as f:
        f.write(json.dumps(split_info, indent=2))


def main():
    file_path_list = load_whitelist()

    directory_name = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")

    try:
        os.makedirs(directory_name)
    except OSError as e:
        print(e.message)
        exit()

    model_filename = os.path.join(directory_name, "model.pkl")

    logger = logging.getLogger("TrainingLogger")
    logger.setLevel(logging.DEBUG)
    log_file_name = os.path.join(directory_name, "training.log")

    fh = logging.FileHandler(log_file_name)
    fh.setLevel(logging.NOTSET)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('[%(asctime)s] - %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("Start time: {}".format(datetime.datetime.now()))
    logger.debug(str(len(file_path_list)) + " file names loaded.")

    split_index = int(len(file_path_list) * 0.9)
    train_file_list = file_path_list[:split_index]
    test_file_list = file_path_list[split_index:]

    save_test_train_split(directory_name, train_file_list, test_file_list)

    train_dataset = BVHRNNDataset(train_file_list, RNN_SEQUENCE_SIZE)
    test_dataset = BVHRNNDataset(test_file_list, RNN_SEQUENCE_SIZE)
    logger.debug("Data has been indexed.")

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=2 ** 12, shuffle=True, num_workers=4)

    plotter = Plotter()

    rnn = RNN(NN_INPUT_SIZE, RNN_RECURRENT_SIZE, NN_OUTPUT_SIZE)

    criterion = nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)

    logger.debug("Starting training.")

    for i in range(10):
        logging.debug("Start of epoch {}".format(i + 1))
        start = time.time()
        adjust_learning_rate(optimizer, i + 1)

        for input_output_sequence in train_loader:
            num_sequences = input_output_sequence[0][0].size()[0]

            recurrent = rnn.init_recurrent(num_sequences)

            for input_value, observed_output_value in input_output_sequence:
                optimizer.zero_grad()
                input_value = Variable(input_value).float()
                observed_output_value = Variable(observed_output_value).float()

                predicted_value, recurrent = rnn(input_value, recurrent)

                loss = criterion(predicted_value, observed_output_value)
                loss.backward(retain_variables=True)
            optimizer.step()

        # Evaluate model and add to plot.
        train_loss = evaluate_rnn_model_on_dataset(rnn, criterion, train_dataset)
        test_loss = evaluate_rnn_model_on_dataset(rnn, criterion, test_dataset)
        plotter.record_value(LOSS_TITLE, "Training", train_loss)
        plotter.record_value(LOSS_TITLE, "Testing", test_loss)

        t_delta = time.time() - start

        plotter.record_value(ITERATION_TITLE, "Iterations", t_delta)

        logger.info("Training loss: {}\tTesting loss:{}".format(train_loss, test_loss))
        logger.info("Time for iteration {}".format(t_delta))

    logger.debug("Completed training...")
    logger.info("End time: {}".format(datetime.datetime.now()))
    logger.debug("Saving model to {}".format(model_filename))
    torch.save(rnn, model_filename)

    train_loss = evaluate_rnn_model_on_dataset(rnn, criterion, train_dataset)
    test_loss = evaluate_rnn_model_on_dataset(rnn, criterion, test_dataset)
    logger.info("Training loss: {}\tTesting loss:{}".format(train_loss, test_loss))

    plotter.prepare_plots()
    plotter.save_plots(directory_name)
    plotter.show_plot()


def adjust_learning_rate(optimizer, epoch):
    lr = (0.01 * (0.6 ** (epoch // 3)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
