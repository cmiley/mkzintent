import logging
import pickle
import torch.utils.data as data
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from matplotlib import rc

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


def evaluate_model_on_dataset(model, criterion, dataset):
    loader = data.DataLoader(dataset=dataset, batch_size=2 ** 12, shuffle=True)

    loss_sum = 0.0
    num_samples = 0.0

    for input_value, observed_output_value in loader:
        input_value = Variable(input_value).float()
        observed_output_value = Variable(observed_output_value).float()

        pred = model(input_value)

        loss = criterion(pred, observed_output_value)
        loss_sum += np.average(loss.data.numpy())
        num_samples += 1.0
    return loss_sum / num_samples


def evaluate_rnn_model_on_dataset(model, criterion, dataset):
    loader = data.DataLoader(dataset=dataset, batch_size=2 ** 12, shuffle=True)

    loss_sum = 0.0
    num_samples = 0.0

    for input_output_sequence in loader:
        num_sequences = input_output_sequence[0][0].size()[0]

        model.init_recurrent(num_sequences)

        for input_value, observed_output_value in input_output_sequence:
            input_value = Variable(input_value).float()
            observed_output_value = Variable(observed_output_value).float()

            predicted_value = model(input_value)

            loss = criterion(predicted_value, observed_output_value)
            loss_sum += np.average(loss.data.numpy())
            num_samples += 1.0
    return loss_sum / num_samples


class Plotter:
    def __init__(self):
        self.plot_values = {}

    def show_plot(self):
        plt.show()

    def prepare_plots(self):
        for title in self.plot_values:
            plt.figure(title)
            for name in self.plot_values[title]:
                plot_data = self.plot_values[title][name]
                xticks = range(len(plot_data))
                plt.plot(xticks, plot_data, label=name)

            plt.legend()

    def record_value(self, title, name, value):
        if not title in self.plot_values:
            self.plot_values[title] = {}

        if not name in self.plot_values[title]:
            self.plot_values[title][name] = []

        self.plot_values[title][name].append(value)

    def pickle_plots(self, path):
        figs = []
        for title in self.plot_values:
            fig = plt.figure(title)
            figs.append(fig)
        print(figs)
        with open(os.path.join(path, "plotting.pickle"), 'wb') as f:
            pickle.dump(figs, f)

    def save_plots(self, path):
        for title in self.plot_values:
            plt.figure(title)
            plt.savefig(os.path.join(path, title + ".png"), bbox_inches='tight')


def adjust_learning_rate(optimizer, epoch):
    # TODO: make these parameters ( potentially make a class )
    lr = (0.01 * (0.6 ** (epoch // 3)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def create_logger(log_name, directory_name):
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    log_file_name = os.path.join(directory_name, log_name + ".log")

    fh = logging.FileHandler(log_file_name)
    fh.setLevel(logging.NOTSET)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('[%(asctime)s] - %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def save_test_train_split(path, train_file_list, test_file_list):
    split_info = {
        "train_file_list": train_file_list,
        "test_file_list": test_file_list
    }
    import json
    with open(os.path.join(path, "test_train_split_info"), "w+") as f:
        f.write(json.dumps(split_info, indent=2))

