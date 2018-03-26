import torch.nn as nn

import time
import datetime

from BVHReader import *
from ModelEvaluationTools import *

LOSS_TITLE = "Loss Graph"
ITERATION_TITLE = "Iteration Graph"


class RNN(nn.Module):
    def __init__(self, input_size, recurrent_size, output_size):
        super(RNN, self).__init__()

        self.recurrent_size = recurrent_size

        self.recurrent_layer = nn.Linear(input_size + recurrent_size, recurrent_size)

        self.recurrent = None
        self.init_recurrent(recurrent_size)

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

    def forward(self, input_value):
        combined = torch.cat((input_value, self.recurrent), 1)

        self.recurrent = self.recurrent_layer(combined)
        output = self.feed_forward_layers(combined)
        return output

    def init_recurrent(self, n):
        self.recurrent = Variable(torch.zeros(n, self.recurrent_size))

    def m_train(self, input_output_sequence, optimizer, criterion):
        num_sequences = input_output_sequence[0][0].size()[0]
        self.init_recurrent(num_sequences)

        for input_value, observed_output_value in input_output_sequence:
            optimizer.zero_grad()
            input_value = Variable(input_value).float()
            observed_output_value = Variable(observed_output_value).float()

            predicted_value = self(input_value)

            loss = criterion(predicted_value, observed_output_value)
            loss.backward(retain_variables=True)
        optimizer.step()


def main():
    file_path_list = load_whitelist()[:5]

    directory_name = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")

    try:
        os.makedirs(directory_name)
    except OSError as e:
        print(e.message)
        print(e.strerror)
        exit()

    model_filename = os.path.join(directory_name, "model.pkl")

    logger = create_logger(directory_name)

    logger.info("Start time: {}".format(datetime.datetime.now()))
    logger.debug(str(len(file_path_list)) + " file names loaded.")

    split_index = int(len(file_path_list) * 0.9)
    train_file_list = file_path_list[:split_index]
    test_file_list = file_path_list[split_index:]

    save_test_train_split(directory_name, train_file_list, test_file_list)

    train_dataset = BVHRNNDataset(train_file_list, conf.RNN_SEQUENCE_SIZE)
    test_dataset = BVHRNNDataset(test_file_list, conf.RNN_SEQUENCE_SIZE)
    logger.debug("Data has been indexed.")

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=2 ** 12, shuffle=True, num_workers=4)

    plotter = Plotter()

    rnn = RNN(conf.NN_INPUT_SIZE, conf.RNN_RECURRENT_SIZE, conf.NN_OUTPUT_SIZE)

    criterion = nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)

    logger.debug("Starting training.")

    for i in range(2):
        logging.info("Start of epoch {}".format(i + 1))
        start = time.time()
        adjust_learning_rate(optimizer, i + 1)

        for input_output_sequence in train_loader:
            rnn.m_train(input_output_sequence, optimizer, criterion)

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
    plotter.pickle_plots(directory_name)
    plotter.save_plots(directory_name)
    plotter.show_plot()


if __name__ == '__main__':
    main()
