import torch.utils.data as data
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

def evaluate_model_on_dataset(model, criterion, dataset):

    loader = data.DataLoader(dataset=dataset, batch_size=2**12, shuffle=True)

    loss_sum = 0.0
    num_samples = 0.0
    for input_value, observed_output_value in loader:
        input_value = Variable(input_value).float()
        observed_output_value = Variable(observed_output_value).float()

        pred = model(input_value)
        loss = criterion(pred, observed_output_value)
        loss_sum += np.average(loss.data.numpy())
        num_samples += 1.0
    return loss_sum/num_samples