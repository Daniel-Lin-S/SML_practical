import os
import logging
import torch
from sklearn.metrics import accuracy_score

"""A training function for a simple neural network."""


# Set up logging
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def train(model, train_loader, optimizer, loss_function, device='mps'):
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
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        
        # logging.info(f"Loss: {loss.item()}")
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate(model, val_loader, loss_function, device='cpu'):
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
        for data, target in val_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            
            # Calculate loss
            loss = loss_function(output, target)
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(output, 1)
            correct = (predicted == target).sum().item()
            
            total_accuracy += correct / target.size(0)
            
    avg_loss = total_loss / len(val_loader)
    avg_accuracy = total_accuracy / len(val_loader)
    
    return avg_loss, avg_accuracy



def run_training_loop(num_epochs, 
                      train_loader,
                      val_loader,
                      model,
                      optimizer,
                      loss_function,
                      model_name='model.pth',
                      save_path=None,
                      device='mps',
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
    
    best_val_loss = float('inf')
    
    if save_path is None:
        save_path = os.path.join('models')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        logging.info(f"Path not provided, saving model to {save_path}")
    else:
        save_path = os.path.join('models', save_path)
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        logging.info(f"Saving model to {save_path}")
    
    model.to(device)  # Move model to device
    
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, loss_function, device)
        val_loss, val_acc = validate(model, val_loader, loss_function, device)
        
        if verbose and epoch % 5 == 0:
            logging.info(f'Epoch: {epoch+1}, train Loss: {train_loss:.4f}, val Loss: {val_loss:.4f}, val Accuracy: {val_acc*100:.2f}%')
            
            if val_loss < best_val_loss:
                logging.info("Saving model...")
                torch.save(model.state_dict(), os.path.join(save_path, model_name))
                best_val_loss = val_loss
            
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    
    logging.info(f"Training complete. Model saved to {save_path}")
    
    logs_dir = "logs"
    
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
        
    train_log_path = os.path.join(logs_dir, "train_log.txt")
    val_log_path = os.path.join(logs_dir, "val_log.txt")
    
    with open(train_log_path, 'w') as f:
        for item in train_losses:
            f.write("%s\n" % item)
            
    with open(val_log_path, 'w') as f:
        for item in val_losses:
            f.write("%s\n" % item)
        
        
    return train_losses, val_losses