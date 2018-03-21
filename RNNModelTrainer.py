import sys
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
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input_value, hidden):
        combined = torch.cat((input_value, hidden), 1)

        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self, n):
        return Variable(torch.zeros(n, self.hidden_size))


def main():
    file_path_list = load_whitelist()

    # TODO: argparse
    model_filename = "rnn-model.pkl"

    logger = logging.getLogger("TrainingLogger")
    logger.setLevel(logging.DEBUG)
    log_file_name = LOG_DIR + datetime.datetime.now().strftime("%Y-%m-%d_%H%M") + ".log"

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

    train_dataset = BVHRNNDataset(train_file_list, RNN_SEQUENCE_SIZE)
    test_dataset = BVHRNNDataset(test_file_list, RNN_SEQUENCE_SIZE)
    logger.debug("Data has been indexed.")

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=2 ** 12, shuffle=True, num_workers=4)

    plotter = Plotter()

    rnn = RNN(NN_INPUT_SIZE, RNN_HIDDEN_SIZE, NN_OUTPUT_SIZE)

    criterion = nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)

    logger.debug("Starting training.")

    for i in range(50):
        logging.debug("Start of epoch {}".format(i + 1))
        start = time.time()
        adjust_learning_rate(optimizer, i + 1)

        for input_output_sequence in train_loader:
            num_sequences = input_output_sequence[0][0].size()[0]

            hidden = rnn.init_hidden(num_sequences)

            for input_value, observed_output_value in input_output_sequence:
                optimizer.zero_grad()
                input_value = Variable(input_value).float()
                observed_output_value = Variable(observed_output_value).float()

                predicted_value, hidden = rnn(input_value, hidden)

                loss = criterion(predicted_value, observed_output_value)
                loss.backward(retain_variables=True)
            optimizer.step()

        # Evaluate model and add to plot.
        train_loss = evaluate_rnn_model_on_dataset(rnn, criterion, train_dataset)
        test_loss = evaluate_rnn_model_on_dataset(rnn, criterion, test_dataset)
        plotter.record_value(LOSS_TITLE, "Training", train_loss)
        plotter.record_value(LOSS_TITLE, "Testing", test_loss)

        t_delta = time.time()-start

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

    plotter.show_plot()


def adjust_learning_rate(optimizer, epoch):
    lr = (0.01 ** ((epoch + 3) // 3))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
