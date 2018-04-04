import torch.nn as nn

import time
import datetime

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
        nn.DataParallel(self.model)

    def forward(self, input_value):
        return self.model(input_value)

    def m_train(self, input_output_pairs, optimizer, criterion):
        input_value, observed_output_value = input_output_pairs
        optimizer.zero_grad()

        input_value = Variable(input_value).float().cuda()
        observed_output_value = Variable(observed_output_value).float().cuda()

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
        print("ERROR:"+ e.message + " " + e.strerror)
        exit()

    model_file_path = os.path.join(directory_name, "model.pkl")

    logger = create_logger(directory_name)

    start_time = datetime.datetime.now()

    logger.info("Start time: {}".format(start_time))
    logger.debug(str(len(file_path_list)) + " file names loaded.")

    split_index = int(len(file_path_list) * 0.9)
    train_file_list = file_path_list[:split_index]
    test_file_list = file_path_list[split_index:]

    train_dataset = BVHDatasetDeltas(train_file_list)
    test_dataset = BVHDatasetDeltas(test_file_list)
    logger.debug("Data has been indexed.")

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=2 ** 12, shuffle=True, num_workers=4)
    test_eval_loader = data.DataLoader(dataset=test_dataset, batch_size=2**12, shuffle=True, num_workers=4)
    train_eval_loader = data.DataLoader(dataset=train_dataset, batch_size=2**12, shuffle=True, num_workers=4)

    plotter = Plotter()
    model = FeedForwardNet(NN_INPUT_SIZE+3, NN_OUTPUT_SIZE)
    model.cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    logger.debug("Starting training.")

    current_avg = 0

    num_epochs = 2
    for epoch in range(num_epochs):
        logging.debug("Start of epoch {}".format(epoch + 1))
        start = time.time()
        adjust_learning_rate(optimizer, epoch + 1)
        for input_output_pair in train_loader:
            model.m_train(input_output_pair, optimizer, criterion)

        logger.info("Evaluating Loss")
        # Evaluate model and add to plot.
        train_loss = evaluate_model_on_dataset(model, criterion, train_eval_loader)
        test_loss = evaluate_model_on_dataset(model, criterion, test_eval_loader)
        plotter.record_value(LOSS_TITLE, "Training", train_loss)
        plotter.record_value(LOSS_TITLE, "Testing", test_loss)

        t_delta = time.time() - start

        plotter.record_value(ITERATION_TITLE, "Iterations", t_delta)

        logger.info("Training loss: {}\tTesting loss:{}".format(train_loss, test_loss))
        logger.info("Time for iteration {}".format(t_delta))

        current_avg = ((current_avg*epoch) + t_delta)/(epoch+1)
        est_end_time = datetime.datetime.now() + datetime.timedelta(seconds=current_avg*(num_epochs-epoch))

        logger.info("Estimated time to finish: {}".format(est_end_time))

    logger.debug("Completed training...")
    logger.info("End time: {}".format(datetime.datetime.now()))

    logger.debug("Saving model to {}".format(model_file_path))
    torch.save(model, model_file_path)

    train_loss = evaluate_model_on_dataset(model, criterion, train_eval_loader)
    test_loss = evaluate_model_on_dataset(model, criterion, test_eval_loader)
    logger.info("Training loss: {}\tTesting loss:{}".format(train_loss, test_loss))

    plotter.prepare_plots()
    plotter.pickle_plots(directory_name)
    plotter.save_plots(directory_name)
    plotter.show_plot()


if __name__ == '__main__':
    main()
