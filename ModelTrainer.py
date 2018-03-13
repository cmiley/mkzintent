import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from torch.autograd import Variable

import time
import datetime

from BVHReader import *
from ModelEvaluationTools import *


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = (0.01 ** ((epoch + 3) // 3))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    file_path_list = load_whitelist()[:50]

    # TODO: argparse
    model_filename = "model.pkl"

    print("Start time: {}".format(datetime.datetime.now()))
    print(str(len(file_path_list)) + " filenames loaded.")

    split_index = int(len(file_path_list) * 0.9)
    train_filelist = file_path_list[:split_index]
    test_filelist = file_path_list[split_index:]

    train_dataset = BVHDataset(train_filelist)
    test_dataset = BVHDataset(test_filelist)
    print("Data has been indexed.")

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=2 ** 12, shuffle=True, num_workers=4)

    plotter = Plotter()
    LOSS_TITLE = "Loss Graph"

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

    print("Starting training.")

    for i in range(10):
        print("Start of epoch {}".format(i + 1))
        start = time.time()
        adjust_learning_rate(optimizer, i + 1)
        for input_value, observed_output_value in train_loader:
            optimizer.zero_grad()

            input_value = Variable(input_value).float()
            observed_output_value = Variable(observed_output_value).float()

            pred = model(input_value)
            loss = criterion(pred, observed_output_value)
            loss.backward()
            optimizer.step()

        # Evaluate model and add to plot.
        train_loss = evaluate_model_on_dataset(model, criterion, train_dataset)
        test_loss = evaluate_model_on_dataset(model, criterion, test_dataset)
        plotter.record_value(LOSS_TITLE, "Training", train_loss)
        plotter.record_value(LOSS_TITLE, "Testing", test_loss)

        print("Training loss: {}\tTesting loss:{}".format(train_loss, test_loss))
        print("Time for iteration {}".format(time.time() - start))

    print("Completed training...")
    print("End time: {}".format(datetime.datetime.now()))
    print("Saving model to {}".format(model_filename))
    torch.save(model, model_filename)

    train_loss = evaluate_model_on_dataset(model, criterion, train_dataset)
    test_loss = evaluate_model_on_dataset(model, criterion, test_dataset)
    print("Training loss: {}\tTesting loss:{}".format(train_loss, test_loss))

    plotter.show_plot()


if __name__ == '__main__':
    main()
