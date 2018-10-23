"""Perceptron."""
import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from torch.autograd import Variable
from torch import optim
from tqdm import tqdm

device = torch.device('cuda:1')


class Perceptron(nn.Module):
    """Simple MLP."""

    def __init__(self, in_dim, hid_dim, out_dim):
        """Initialize the perceptron."""
        super(Perceptron, self).__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        """Do a forward pass."""
        x = self.sigmoid(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

    def hidden(self, x):
        """Get the hidden response to a forward pass."""
        return self.sigmoid(F.linear(x,
                                     self.fc1.weight,
                                     self.fc1.bias))

    def run(self, X):
        """Run some numpy data through the model."""
        data = Variable(torch.from_numpy(X.astype('float32')))
        return self.forward(data).data.numpy()


def train(model, num_epoch, X, batch_size):
    """Training function."""
    model.train()
    train_loss = 0

    loss_func = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    y = Variable(torch.arange(X.shape[0]).type(torch.long)).cuda()

    lowest_loss = np.inf
    bad = 0

    for i in tqdm(range(num_epoch), total=num_epoch):

        indices = np.random.permutation(np.arange(X.shape[0]))
        X_ = X[indices]
        X_ = Variable(torch.from_numpy(X_)).cuda()
        y_ = y[indices]

        for x in range(0, X.shape[0], 250):
            prediction = model(X_[x:x+250])

            loss = loss_func(prediction, y_[x:x+250])
            optimizer.zero_grad()
            loss.backward()

            train_loss += loss.data
            optimizer.step()

        if train_loss < lowest_loss:
            lowest_loss = train_loss
            bad = 0
        else:
            bad += 1
        train_loss = 0

        if bad == 20:
            print("Stop after {} epochs".format(i))
            return
