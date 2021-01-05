import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pylab as plt
from torch import optim, nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split

from data_loader import PupilsDatasetLoader
import time

from tester import preview_images
from models.resnet_model import ResNetModel


def benchmark_loading_speed(loader: DataLoader) -> float:
    """
    Utility method that tries to load 100 batches from dataloader and gives an estimate of how fast it runs
    :param loader: any pytorch data loader
    :return: loading speed expressed in samples per second
    """
    i = 0
    i_max = 100
    t_from = time.time()
    for frame, label in loader:
        if i > i_max:
            break
        i+=1
    t_to = time.time()
    return i_max/(t_to-t_from)*loader.batch_size


def loss_batch(loss_func, output, target, optimizer=None):
    loss = loss_func(output, target)
    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()


def loss_epoch(model, loss_function, data_loader, optimizer=None):
    running_loss=0.0
    epoch_length=100
    i = 0
    for xb, yb in data_loader:
        i += 1
        yb = yb.type(torch.float32).to(device)
        output = model(xb.to(device))
        loss_b = loss_batch(loss_function, output, yb, optimizer)
        running_loss += loss_b

        loss = running_loss / float(epoch_length)
        if i > epoch_length:
            return loss


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(model, params):
        # extract model parameters
        num_epochs = params["num_epochs"]
        loss_func = params["loss_func"]
        optimizer = params["optimizer"]
        train_dl = params["train_dl"]
        val_dl = params["val_dl"]
        lr_scheduler = params["lr_scheduler"]

        # history of loss values in each epoch
        loss_history = {
            "train": [],
            "val": [],
        }

        # a deep copy of weights for the best performing model
        best_model_wts = copy.deepcopy(model.state_dict())

        # initialize best loss to a large value
        best_loss = float('inf')

        # main loop
        for epoch in range(num_epochs):
            # get current learning rate
            current_lr = get_learning_rate(optimizer)
            print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs - 1, current_lr))

            # train model on training dataset
            model.train()
            train_loss = loss_epoch(model, loss_func, train_dl, optimizer)

            # collect loss and metric for training dataset
            loss_history["train"].append(train_loss)

            # evaluate model on validation dataset
            model.eval()
            with torch.no_grad():
                val_loss = loss_epoch(model, loss_func, val_dl)

            # store best model
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())

                # store weights into a local file
                model.save(Path("trained_models"), "resnet")
                print("Copied best model weights!")

            # collect loss and metric for validation dataset
            loss_history["val"].append(val_loss)

            # learning rate schedule
            lr_scheduler.step(val_loss)
            if current_lr != get_learning_rate(optimizer):
                print("Loading best model weights!")
                model.load_state_dict(best_model_wts)

            print("train loss: %.6f, dev loss: %.6f" % (train_loss, val_loss))
            print("-" * 10)

        # load best model weights
        model.load_state_dict(best_model_wts)

        return model, loss_history


if __name__ == '__main__':
    # Reproducibility is important!
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    parser = argparse.ArgumentParser(
        description="Trains selected model given the training config and folder with training images and labels."
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the training config json file"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the dataset with jpg and txt files (samples and labels)"
    )
    args = parser.parse_args()

    with open(args.config_path, "r") as fp:
        training_config = json.load(fp)

    dataset = PupilsDatasetLoader(Path(args.dataset_path), training_config['augmentations'])
    train_dataset, validation_dataset = random_split(dataset, lengths=[int(0.8*len(dataset)), int(0.2*len(dataset))])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=8, shuffle=False)

    if torch.cuda.is_available():
        print("Using CUDA")
        device = torch.device("cuda")
    else:
        print("Cuda not available, using CPU")
        device = torch.device("cpu")

    #preview_images(train_loader)

    # create model
    model = ResNetModel(training_config['hyperparameters']).to(device)

    loss_func = nn.SmoothL1Loss(reduction="sum",)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200, verbose=True)

    params_train = {
        "num_epochs": 600,
        "optimizer": optimizer,
        "loss_func": loss_func,
        "train_dl": train_loader,
        "val_dl": validation_loader,
        "lr_scheduler": lr_scheduler,
    }

    # Train-Validation Progress
    num_epochs = params_train["num_epochs"]

    # train and validate the model
    cnn_model, loss_hist = train(model, params_train)

    # plot loss progress
    plt.title("Train-Val Loss")
    plt.plot(range(1, num_epochs + 1), loss_hist["train"], label="train")
    plt.plot(range(1, num_epochs + 1), loss_hist["val"], label="val")
    plt.ylabel("Loss")
    plt.xlabel("Training Epochs")
    plt.legend()
    plt.show()