import torch.utils.data as data
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt


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

        recurrent = model.init_recurrent(num_sequences)

        for input_value, observed_output_value in input_output_sequence:
            input_value = Variable(input_value).float()
            observed_output_value = Variable(observed_output_value).float()

            predicted_value, recurrent = model(input_value, recurrent)

            loss = criterion(predicted_value, observed_output_value)
            loss_sum += np.average(loss.data.numpy())
            num_samples += 1.0
    return loss_sum / num_samples


class Plotter:
    def __init__(self):
        self.plot_values = {}

    def show_plot(self):
        for title in self.plot_values:
            plt.figure(title)
            for name in self.plot_values[title]:
                plot_data = self.plot_values[title][name]
                xticks = range(len(plot_data))
                plt.plot(xticks, plot_data, label=name)

            plt.legend()
        plt.show()

    def record_value(self, title, name, value):
        if not title in self.plot_values:
            self.plot_values[title] = {}

        if not name in self.plot_values[title]:
            self.plot_values[title][name] = []

        self.plot_values[title][name].append(value)
