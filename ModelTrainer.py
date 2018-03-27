import sys

import os
import torch.nn as nn

import time
import datetime
import logging

from BVHReader import *
from ModelEvaluationTools import *

LOSS_TITLE = "Loss Graph"
ITERATION_TITLE = "Iteration Graph"


class FeedForwardNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(FeedForwardNet, self).__init__()

        # Hidden layer size
        h1, h2, h3 = 200, 150, 50
        self.model = torch.nn.Sequential(
            nn.Linear(input_size, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Linear(h3, output_size)
        )

    def forward(self, input_value):
        return self.model(input_value)

    def m_train(self, input_output_pairs, optimizer, criterion):
        input_value, observed_output_value = input_output_pairs
        optimizer.zero_grad()

        input_value = Variable(input_value).float()
        observed_output_value = Variable(observed_output_value).float()

        predicted_value = self(input_value)
        loss = criterion(predicted_value, observed_output_value)
        loss.backward()
        optimizer.step()


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

    logger = create_logger(directory_name)

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
    model = FeedForwardNet(NN_INPUT_SIZE, NN_OUTPUT_SIZE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    logger.debug("Starting training.")

    num_epochs = 10
    for epoch in range(num_epochs):
        logging.debug("Start of epoch {}".format(epoch + 1))
        start = time.time()
        adjust_learning_rate(optimizer, epoch + 1)
        for input_output_pair in train_loader:
            model.m_train(input_output_pair, optimizer, criterion)

        # Evaluate model and add to plot.
        train_loss = evaluate_model_on_dataset(model, criterion, train_dataset)
        test_loss = evaluate_model_on_dataset(model, criterion, test_dataset)
        plotter.record_value(LOSS_TITLE, "Training", train_loss)
        plotter.record_value(LOSS_TITLE, "Testing", test_loss)

        t_delta = time.time() - start

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

    plotter.prepare_plots()
    plotter.pickle_plots(directory_name)
    plotter.save_plots(directory_name)
    plotter.show_plot()


if __name__ == '__main__':
    main()
