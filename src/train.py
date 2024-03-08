import sys
import os


src_dir = os.path.join(os.getcwd(), '../src')
if src_dir not in sys.path:
    sys.path.append(src_dir)

from nn_model import *
from run_nn import *
from train_nn import *
from dataloader import *

import pandas as pd

print(os.getcwd())


# prepare mnist
from torchvision import datasets, transforms
mnist_train = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())

train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
val_loader = DataLoader(mnist_test, batch_size=64, shuffle=True)

mlp = MLP(input_dim=28*28,
            hidden_dims=[32, 16],
            output_dim=10,
            dropout=0.0)

train_loss, test_loss = run_training_loop(
    num_epochs=100,
    train_loader=train_loader,
    val_loader=val_loader,
    model=mlp,
    optimizer=torch.optim.Adam(mlp.parameters(), lr=0.1),
    loss_function=torch.nn.CrossEntropyLoss(),
    device="mps"
)
