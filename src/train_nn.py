import os
import logging
import torch

"""A training function for a simple neural network."""



# Set up logging
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def train(model, train_loader, optimizer, loss_function):
    """
    Trains the neural network model using the given data and optimizer.
    
    Args:
        - model: The neural network model
        - train_loader: DataLoader for training data
        - optimizer: Optimization algorithm in torch.optim
        - loss_function: Loss function
        
    Returns:
        The average loss over the training data.
    """
    model.train()
    total_loss = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate(model, val_loader, loss_function):
    """
    Evaluates the neural network model using the given data.
    
    Args:
        - model: The neural network model
        - val_loader: DataLoader for validation data
        - loss_function: Loss function
        
    Returns:
        The average loss over the validation data.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            loss = loss_function(output, target)
            total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    return avg_loss


def run_training_loop(num_epochs, 
                      train_loader,
                      val_loader,
                      model,
                      optimizer,
                      loss_function,
                      model_name='model.pth',
                      save_path=None,
                      verbose=True):
    """
    Runs the training loop for a given number of epochs.
    Saves the trained model to a given path and writes logs to text files.
    
    Args:  
        - num_epochs: Number of epochs to train for
        - train_loader: DataLoader for training data
        - val_loader: DataLoader for validation data
        - model: Neural network model
        - optimizer: Optimization algorithm in torch.optim
        - loss_function: Loss function
        - save_path: Path to save the trained model
        - verbose: Whether to print training progress 
        
    Returns:  
        - list1: Training losses
        - list2: Validation losses 
    """
    
    # Set up logging
    train_losses = []
    val_losses = []
    
    if save_path is None:
        save_path = os.path.join('models', model_name)
        logging.info(f"Path not provided, saving model to {save_path}")
    else:
        save_path = os.path.join('models', save_path, model_name)
        logging.info(f"Saving model to {save_path}")
    
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, loss_function)
        val_loss = validate(model, val_loader, loss_function)
        
        if verbose and epoch % 5 == 0:
            logging.info(f'Epoch: {epoch+1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
            # save model
            torch.save(model.state_dict(), save_path)
            
    logging.info(f"Training complete. Model saved to {save_path}")
    
    # write logs to text file
    with open(f"logs/{model_name}_train_logs.txt", "w") as file:
        file.write(f"Training Losses: {train_losses}\n")
    
    with open(f"logs/{model_name}_val_logs.txt", "w") as file:
        file.write(f"Validation Losses: {val_losses}\n")
        

    return train_losses, val_losses