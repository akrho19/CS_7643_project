import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model1 import Model1
from model_simple import ModelSimple
from endovis_dataset import EndovisDataset
from reformat_data import convert_all_data


def train_segmentation():
    segmentation_model_weights_path = "model1_weights.pth"
    tracking_model_weights_path = "tracking_model_weights.pth"

    og_data_path = r"original_data"
    data_path = r"data"
    
    batch_size = 16
    learning_rate = 0.0001
    momentum = 0.9
    epochs = 10

    # Model
    model = ModelSimple() # Switch out with whichever model you like

    folder = os.path.join(data_path, "segmentation_train")
    dataset = EndovisDataset(folder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) # TODO can tune batch size

    criterion = nn.BCELoss() # Because https://medium.com/@kitkat73275/multi-label-classification-8d8ae55e8373#:~:text=For%20multi%2Dlabel%20classification%20with,class%20labels%20for%20new%20instances.
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum) # TODO tune these hyperparameters

    for epoch in range(epochs):
        print("Epoch %d:" % epoch)
        for frames, truths in dataloader:
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, truths)
            loss.backward()
            optimizer.step()
            print("    Loss: %f" % loss.item())

    torch.save(model.state_dict(), segmentation_model_weights_path)


if __name__ == "__main__":
    train_segmentation()
