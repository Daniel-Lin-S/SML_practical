# """Run training."""

# import os
# import argparse
# import logging
# import torch
# import numpy as np
# from dataloader import *


# from nn_model import *
# from train_nn import *


# def main(batch_size, learning_rate, hidden_dim):
#     """Run training."""
#     # Set up logging
#     logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

#     # Set up device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     logging.info(f"Using device: {device}")

#     # Load data
#     X_train = np.load("data/X_train.npy")
#     y_train = np.load("data/y_train.npy")
#     X_val = np.load("data/X_val.npy")
#     y_val = np.load("data/y_val.npy")

#     # Create DataLoader
#     train_loader = MusicData((X_train, y_train), batch_size=32, shuffle=True)
#     val_loader = MusicData((X_val, y_val), batch_size=32)

#     # Create model
#     model = MLP(input_dim=264, hidden_dim=100, output_dim=10).to(device)

#     # Create optimizer
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#     # Create loss function
#     loss_function = torch.nn.CrossEntropyLoss()

#     # Train
#     for epoch in range(10):
#         train_loss = train(model, train_loader, optimizer, loss_function)
#         val_loss = validate(model, val_loader, loss_function)
#         logging.info(f"Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}")


