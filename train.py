#!/usr/bin/env python

import matplotlib.pyplot
import logging
import os
import time
import string
import numpy as np
import torch
import torchvision.transforms.functional as TF
import torch.utils.data as utils
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from definitions import EMNIST_TRAIN_PATH, EMNIST_TEST_PATH, SAVED_NETWORK_PATH
from load_image import EmnistLoader
from cnn import ConvNet

# Hyper parameters
num_epochs = 5
num_classes = 37
batch_size = 5
learning_rate = 0.001
momentum = 0.15
classes = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'
network_name = 'emnist_network'

# Get emnist images
emnist_loader = EmnistLoader()
emnist_loader.load()

# Set up logging
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

# Set matplotlib backend and turn on interactive mode
#matplotlib.use("QT5Agg")
#matplotlib.pyplot.ion()
#matplotlib.pyplot.ylabel("Loss")
#matplotlib.pyplot.show()

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Get training data and transform into dataset
train_data = torch.stack([torch.tensor(i, dtype=torch.float32) for i in emnist_loader.train_data]) # transform to torch tensors
train_labels = torch.stack([torch.tensor(i, dtype=torch.long) for i in emnist_loader.train_labels])

train_data = torch.stack([i.unsqueeze(0) for i in train_data])
train_data = train_data.type(torch.cuda.FloatTensor)

# Create your dataset and dataloader
train_dataset = utils.TensorDataset(train_data, train_labels)
train_loader = utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Get testing data and transform into dataset
test_data = torch.stack([torch.tensor(i, dtype=torch.float32) for i in emnist_loader.test_data])
test_labels = torch.stack([torch.tensor(i, dtype=torch.long) for i in emnist_loader.test_labels])

test_data = torch.stack([i.unsqueeze(0) for i in test_data])
test_data = test_data.type(torch.cuda.FloatTensor)

# Create your dataset and dataloader
test_dataset = utils.TensorDataset(test_data, test_labels)
test_loader = utils.DataLoader(test_dataset, shuffle=True)

model_file = Path(os.path.join(SAVED_NETWORK_PATH, network_name))
if model_file.is_file():
    model = torch.load(model_file)
    model.eval()
else:
    # Create your network
    model = ConvNet(num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (t_i, t_l) in enumerate(train_loader):
            t_i = t_i.to(device=device)
            t_l = t_l.to(device=device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(t_i)
            loss = F.nll_loss(outputs, t_l)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    torch.save(model, os.path.join(SAVED_NETWORK_PATH, network_name))

total_step = len(test_loader)
test_loss = 0
correct = 0
with torch.no_grad():
    for i, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)

        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        #imgplot = plt.imshow(TF.to_pil_image(model.final_layer))
        #plt.show()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
