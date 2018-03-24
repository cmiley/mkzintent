import sys

import os
import torch.nn as nn

import time
import datetime
import logging

from BVHReader import *
from ModelEvaluationTools import *


LOSS_LOG_LEVEL = logging.INFO
TIMING_LOG_LEVEL = logging.INFO
IO_LOG_LEVEL = logging.NOTSET

LOSS_TITLE = "Loss Graph"
ITERATION_TITLE = "Iteration Graph"


def main():
    file_path_list = load_whitelist()[:5]

    # TODO: argparse
    directory_name = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")

    try:
        os.makedirs(directory_name)
    except OSError as e:
        print(e.message)
        exit()

    model_file_path = os.path.join(directory_name, "model.pkl")

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

    train_dataset = BVHDataset(train_file_list)
    test_dataset = BVHDataset(test_file_list)
    logger.debug("Data has been indexed.")

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=2 ** 12, shuffle=True, num_workers=4)

    plotter = Plotter()

    # Hidden layer size
    h1, h2, h3 = 200, 150, 50
    model = torch.nn.Sequential(
        nn.Linear(NN_INPUT_SIZE, h1),
        nn.ReLU(),
        nn.Linear(h1, h2),
        nn.ReLU(),
        nn.Linear(h2, h3),
        nn.ReLU(),
        nn.Linear(h3, NN_OUTPUT_SIZE)
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    logger.debug("Starting training.")

    for i in range(10):
        logging.debug("Start of epoch {}".format(i + 1))
        start = time.time()
        adjust_learning_rate(optimizer, i + 1)
        for input_value, observed_output_value in train_loader:
            optimizer.zero_grad()

            input_value = Variable(input_value).float()
            observed_output_value = Variable(observed_output_value).float()

            predicted_value = model(input_value)
            loss = criterion(predicted_value, observed_output_value)
            loss.backward()
            optimizer.step()

        # Evaluate model and add to plot.
        train_loss = evaluate_model_on_dataset(model, criterion, train_dataset)
        test_loss = evaluate_model_on_dataset(model, criterion, test_dataset)
        plotter.record_value(LOSS_TITLE, "Training", train_loss)
        plotter.record_value(LOSS_TITLE, "Testing", test_loss)

        t_delta = time.time()-start

        plotter.record_value(ITERATION_TITLE, "Iterations", t_delta)

        logger.info("Training loss: {}\tTesting loss:{}".format(train_loss, test_loss))
        logger.info("Time for iteration {}".format(t_delta))

    logger.debug("Completed training...")
    logger.info("End time: {}".format(datetime.datetime.now()))
    logger.debug("Saving model to {}".format(model_file_path))
    torch.save(model, model_file_path)

    train_loss = evaluate_model_on_dataset(model, criterion, train_dataset)
    test_loss = evaluate_model_on_dataset(model, criterion, test_dataset)
    logger.info("Training loss: {}\tTesting loss:{}".format(train_loss, test_loss))

    plotter.show_plot()


def adjust_learning_rate(optimizer, epoch):
    lr = (0.01 ** ((epoch + 3) // 3))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
