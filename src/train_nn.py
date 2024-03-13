"""Training function for a simple neural network."""

import os
import time
import argparse
import logging
import torch
from sklearn.metrics import accuracy_score

"""A training function for a simple neural network."""


# Set up logging
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def train(model, train_loader, optimizer, loss_function, scheduler=None, device="mps"):
    """
    Trains the neural network model using the given data and optimizer.

    Args:
        - model: The neural network model
        - train_loader: DataLoader for training data
        - optimizer: Optimization algorithm in torch.optim
        - loss_function: Loss function
        - scheduler: Learning rate scheduler

    Returns:
        The average loss over the training data.
    """
    model.train()
    total_loss = 0
    for X, y in train_loader:
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = loss_function(output, y)

        # logging.info(f"Loss: {loss.item()}")

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Step the scheduler
    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate(model, val_loader, loss_function, device="cpu"):
    """
    Evaluates the neural network model using the given data.

    Args:
        - model: The neural network model
        - val_loader: DataLoader for validation data
        - loss_function: Loss function
        - device: Device to perform computations

    Returns:
        - avg_loss: The average loss over the validation data.
        - avg_accuracy: The average accuracy over the validation data.
    """
    model.eval()
    total_loss = 0
    total_accuracy = 0

    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)
            y = y.to(device)
            output = model(X)

            # Calculate loss
            loss = loss_function(output, y)
            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(output, 1)
            correct = (predicted == y).sum().item()

            total_accuracy += correct / y.size(0)

    avg_loss = total_loss / len(val_loader)
    avg_accuracy = total_accuracy / len(val_loader)

    return avg_loss, avg_accuracy


def run_training_loop(
    num_epochs,
    train_loader,
    val_loader,
    model,
    optimizer,
    loss_function,
    scheduler=None,
    model_name="model.pth",
    save_path=None,
    device="mps",
    verbose=5,
    write_logs=True,
    patience=10,
):
    """
    Runs the training loop for a given number of epochs.
    Saves the trained model to a given path and writes logs to text files.

    Args:
        - num_epochs (int): Number of epochs to train the model
        - train_loader: DataLoader for training data
        - val_loader: DataLoader for validation data
        - model: The neural network model
        - optimizer: Optimization algorithm in torch.optim
        - loss_function: Loss function
        - scheduler: Learning rate scheduler
        - model_name (str): Name of the model file to save
        - save_path (str): Path to save the model
        - device (str): Device to perform computations
        - verbose (int): Print training logs every `verbose` epochs,
                        if is 0, no logs will be printed.
        - write_logs (bool): Write training and validation logs to text files
        - patience (int): Number of epochs to wait before early stopping

    Returns:
        - list1: Training losses
        - list2: Validation losses
    """

    TIME = time.strftime("%Y-%m-%d_%H-%M-%S")

    # Set up logging
    train_losses = []
    val_losses = []
    val_accs = []

    best_val_loss = float("inf")
    best_val_acc = 0

    if save_path is None:
        save_path = os.path.join("models")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        logging.info(f"Path not provided, saving model to {save_path}")
    else:
        save_path = os.path.join("models", save_path)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        logging.info(f"Saving model to {save_path}")

    model.to(device)  # Move model to device

    # increment when no improvement in val_loss
    wait = 0

    for epoch in range(num_epochs):
        train_loss = train(
            model, train_loader, optimizer, loss_function, scheduler, device
        )
        val_loss, val_acc = validate(model, val_loader, loss_function, device)

        if verbose > 0 and epoch % verbose == 0:
            logging.info(
                f"Epoch: {epoch+1}, train Loss: {train_loss:.4f}, val Loss: {val_loss:.4f}, val Accuracy: {val_acc*100:.2f}%"
            )
            # print(f'Epoch: {epoch+1}, train Loss: {train_loss:.4f}, val Loss: {val_loss:.4f}, val Accuracy: {val_acc*100:.2f}%')

        if val_acc > best_val_acc:
            logging.info(f"Saving model with acc {val_acc}")
            torch.save(model.state_dict(), os.path.join(save_path, model_name))
            best_val_loss = val_loss
            best_val_acc = val_acc
            wait = 0
        else:
            wait += 1
            if wait > patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

    logging.info(f"Training complete. Model saved to {save_path}")

    logs_dir = "logs"

    if write_logs:
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)

        train_log_path = os.path.join(
            logs_dir, f"{model_name}_{TIME}_train_loss_log.txt"
        )
        val_log_path = os.path.join(logs_dir, f"{model_name}_{TIME}_val_loss_log.txt")
        val_acc_log_path = os.path.join(
            logs_dir, f"{model_name}_{TIME}_val_acc_log.txt"
        )

        with open(train_log_path, "w") as f:
            for item in train_losses:
                f.write("%s\n" % item)

        with open(val_log_path, "w") as f:
            for item in val_losses:
                f.write("%s\n" % item)

        with open(val_acc_log_path, "w") as f:
            for item in val_accs:
                f.write("%s\n" % item)

    return train_losses, val_losses, val_accs
