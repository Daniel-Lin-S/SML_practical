"""Training function for a simple neural network."""

import os
import time
import argparse
import logging
import torch
import numpy as np
from sklearn.metrics import accuracy_score

"""A training function for a simple neural network."""


# Set up logging
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

class MonotonicAnnealingScheme:
    """Implements the monotonic annealing scheme for beta-VAE."""
    def __init__(self, start_beta, end_beta, total_steps, increasing=True, weight=1e-4):
        self.start_beta = start_beta
        self.end_beta = end_beta
        self.total_steps = total_steps
        self.increasing = increasing
        self.weight = weight
    
    def get_beta(self, current_step):
        progress = current_step / self.total_steps
        if self.increasing:
            beta = self.start_beta + (self.end_beta - self.start_beta) * progress * self.weight
        else:
            beta = self.end_beta - (self.end_beta - self.start_beta) * progress * self.weight
        return beta


class CyclicAnnealingScheme:
    """Implements the cyclic annealing scheme for beta-VAE."""
    def __init__(self, 
                 min_beta=0.0, 
                 max_beta=1.0, 
                 period=50, 
                 total_steps=1000,
                 start_step=0):
        """
        Args:   
            - min_beta (float): The minimum value of beta.
            - max_beta (float): The maximum value of beta.
            - period (int): The period of the cycle.
            - total_steps (int): The total number of steps.
            - start_step (int): The starting step.
        """
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.period = period
        self.total_steps = total_steps
        self.start_step = start_step
    
    def get_beta(self, current_step):
        if current_step < self.start_step:
            return self.min_beta
        cycle = np.floor(1 + (current_step - self.start_step) / (2 * self.period))
        x = np.abs((current_step - self.start_step) / self.period - 2 * cycle + 1)
        beta = self.min_beta + (self.max_beta - self.min_beta) * np.maximum(0, (1 - x))
        return beta





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
        - The average loss over the training data.
        - The average accuracy over the training data.
    """
    model.train()
    total_loss = 0
    total_accuracy = 0
    for X, y in train_loader:
        model.train()
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = loss_function(output, y)

        # logging.info(f"Loss: {loss.item()}")

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        model.eval()
        _, predicted = torch.max(output, 1)
        correct = (predicted == y).sum().item()

        total_accuracy += correct / y.size(0)

    # Step the scheduler
    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / len(train_loader)
    avg_accuracy = total_accuracy / len(train_loader)

    return avg_loss, avg_accuracy


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
    loss_function=None,
    scheduler=None,
    model_name="model.pth",
    save_path=None,
    device="mps",
    verbose=5,
    write_logs=True,
    patience=10,
    train_function=train,
    val_function=validate,
    annealing_scheme="monotonic",
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
        - train_function: Function to train the model
        - val_function: Function to validate the model

    Returns:
        - list1: Training losses
        - list2: Validation losses
    """

    TIME = time.strftime("%Y-%m-%d_%H-%M-%S")

    training_vae = True if model.__class__.__name__ == "BetaVAE" or model.__class__.__name__ == "ConvVAE" else False

    if loss_function is None and not model.__class__.__name__ == "BetaVAE":
        raise ValueError("Loss function is required for training")

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

        if not training_vae:
            train_loss, train_acc = train_function(
                model, train_loader, optimizer, loss_function, scheduler, device
            )
            val_loss, val_acc = val_function(model, val_loader, loss_function, device)

            if verbose > 0 and epoch % verbose == 0:
                logging.info(
                    f"Epoch: {epoch+1}, train Loss: {train_loss:.4f}, val Loss: {val_loss:.4f}, train Acc: {train_acc*100:.2f}%, val Acc: {val_acc*100:.2f}%"
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

        else:
            # Train VAE
            train_loss, train_recon_loss, train_KLD = train_vae(
                model,
                train_loader,
                optimizer,
                annealing_scheme,
                device,
                current_epoch=epoch,
                max_epochs=num_epochs,
            )

            val_loss, val_recon_loss, val_KLD = validate_vae(
                model,
                val_loader,
                device,
                annealing_scheme=annealing_scheme,
                max_epochs=num_epochs,
                current_epoch=epoch,
            )

            if verbose > 0 and epoch % verbose == 0:
                logging.info(
                    f"Epoch: {epoch+1}, train Loss: {train_loss:.4f}, val Loss: {val_loss:.4f}, train Recon Loss: {train_recon_loss:.4f}, val Recon Loss: {val_recon_loss:.4f}, train KLD: {train_KLD:.4f}, val KLD: {val_KLD:.4f}"
                )
                # print(f'Epoch: {epoch+1}, train Loss: {train_loss:.4f}, val Loss: {val_loss:.4f}, val Accuracy: {val_acc*100:.2f}%')

            if val_loss < best_val_loss:
                logging.info(f"Saving model with recon loss {val_recon_loss} and val loss {val_loss}")
                torch.save(model.state_dict(), os.path.join(save_path, model_name))
                best_val_loss = val_loss
                wait = 0

            else:
                wait += 1
                if wait > patience:
                    logging.info(f"Early stopping at epoch {epoch+1}")
                    break

            train_losses.append(train_loss)
            val_losses.append(val_loss)

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



def vae_loss_function(x_true, reconstruction, log_var, mu, beta):
    """Implement the loss function for the VAE."""  
    # print(reconstruction.shape, x_true.shape)
    recons_loss = torch.nn.functional.mse_loss(reconstruction, x_true)
    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
    
    # modify to include beta
    
    loss = recons_loss + beta * kld_loss * 0.001
    return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss}

def train_vae(
    model,
    train_loader,
    optimizer,
    annealing_scheme=None,
    device="mps",
    current_epoch=0,
    max_epochs=1000,
):
    """
    Trains the VAE model using the given data and optimizer.

    Args:
        - model: The VAE model
        - train_loader: DataLoader for training data
        - optimizer: Optimization algorithm in torch.optim
        - scheduler: Learning rate scheduler
        - device: Device to perform computations
        - annealing_scheme: The annealing scheme to use for the KLD loss
        - current_epoch: The current epoch
        - max_epochs: The maximum number of epochs

    Returns:
        - The average loss over the training data.
        - The average reconstruction loss over the training data.
        - The average KLD loss over the training data.
    """
    model.train()
    total_loss = 0
    total_reconstruction_loss = 0
    total_KLD = 0
    if annealing_scheme == "monotonic":
        annealing_scheme = MonotonicAnnealingScheme(
            start_beta=0.001, end_beta=1.0, total_steps=max_epochs, increasing=True
        )
    elif annealing_scheme == "cyclic":
        annealing_scheme = CyclicAnnealingScheme(
            min_beta=0.0, max_beta=1.0, total_steps=max_epochs, start_step=0
        )
    else:
        beta = 1.0
    for X, y in train_loader:
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        
        output = model(X)
        
        if annealing_scheme is not None:
            beta = annealing_scheme.get_beta(current_epoch)
        
        all_loss = vae_loss_function(X, output, model.logvar, model.mu, beta)
        
        loss = all_loss["loss"]
        reconstruction_loss = all_loss["Reconstruction_Loss"]
        KLD = all_loss["KLD"]

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_reconstruction_loss += reconstruction_loss.item()
        total_KLD += KLD.item()

    avg_loss = total_loss / len(train_loader)
    avg_loss_reconstruction = total_reconstruction_loss / len(train_loader)
    avg_loss_KLD = total_KLD / len(train_loader)

    return avg_loss, avg_loss_reconstruction, avg_loss_KLD


def validate_vae(
    model,
    val_loader,
    device="cpu",
    annealing_scheme="monotonic",
    max_epochs=1000,
    current_epoch=0,
):
    """
    Evaluates the VAE model using the given data.

    Args:
        - model: The VAE model
        - val_loader: DataLoader for validation data
        - device: Device to perform computations

    Returns:
        - avg_loss: The average loss over the validation data.
        - avg_loss_reconstruction: The average reconstruction loss over the validation data.
        - avg_loss_KLD: The average KLD loss over the validation data.
    """
    model.eval()
    total_loss = 0
    total_reconstruction_loss = 0
    total_KLD = 0

    if annealing_scheme == "monotonic":
        annealing_scheme = MonotonicAnnealingScheme(
            start_beta=0.0, end_beta=1.0, total_steps=max_epochs, increasing=True
        )
    elif annealing_scheme == "cyclic":
        annealing_scheme = CyclicAnnealingScheme(
            min_beta=0.0, max_beta=1.0, total_steps=max_epochs, start_step=0
        )
    else:
        beta = 1.0

    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)
            y = y.to(device)
            output = model(X)

            if annealing_scheme is not None:
                beta = annealing_scheme.get_beta(current_epoch)

            all_loss = vae_loss_function(X, output, model.logvar, model.mu, beta)
            loss = all_loss["loss"]
            reconstruction_loss = all_loss["Reconstruction_Loss"]
            KLD = all_loss["KLD"]
            total_loss += loss.item()
            total_reconstruction_loss += reconstruction_loss.item()
            total_KLD += KLD.item()

    avg_loss = total_loss / len(val_loader)
    avg_loss_reconstruction = total_reconstruction_loss / len(val_loader)
    avg_loss_KLD = total_KLD / len(val_loader)
    return avg_loss, avg_loss_reconstruction, avg_loss_KLD
