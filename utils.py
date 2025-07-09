import torch
import torch.nn.functional as F


def average_displacement_error(y, y_pred):
    mask = (y != -1).to(y.device)
    if torch.sum(mask) == 0:
        return torch.tensor(0.0, device=y.device)
    result = torch.sum((F.mse_loss(y_pred, y, reduction='none') * mask).sum(dim=2).sqrt()) / (torch.sum(mask)/2)
    return result


def final_displacement_error(y, y_pred):
    """
    Final Displacement Error (FDE): computes the average L2 distance between 
    the last predicted point and ground truth for each trajectory in the batch.
    
    y, y_pred: shape (batch_size, seq_len, 2)
    masked ground truth values are set to -1
    """
    # Mask where ground truth is valid
    valid_mask = (y != -1)  # shape: (B, T, 2)
    valid_steps = valid_mask[:, :, 0]  # shape: (B, T)

    # Get index of last valid timestep per trajectory
    last_indices = valid_steps.sum(dim=1) - 1  # shape: (B,)
    
    # Batch indices
    batch_size = y.size(0)
    batch_indices = torch.arange(batch_size, device=y.device)
    
    # Get final valid ground truth and predicted positions
    final_gt = y[batch_indices, last_indices]       # shape: (B, 2)
    final_pred = y_pred[batch_indices, last_indices]  # shape: (B, 2)

    # Compute Euclidean distance per trajectory
    fde_per_traj = torch.norm(final_pred - final_gt, dim=1)  # shape: (B,)
    
    # Filter out trajectories with no valid steps
    valid_trajectories = last_indices >= 0
    if valid_trajectories.sum() == 0:
        return torch.tensor(0.0, device=y.device)

    return fde_per_traj[valid_trajectories].mean()


def trajectory_accuracy(y, y_pred, threshold=2.0):
    """Percentage of trajectories with ADE below threshold (in pixels)"""
    mask = (y != -1).to(y.device)
    if torch.sum(mask) == 0:
        return torch.tensor(0.0, device=y.device)
    
    # Calculate MSE loss and apply mask
    mse_loss = F.mse_loss(y_pred, y, reduction='none')  # Shape: (batch, seq, 2)
    masked_loss = mse_loss * mask  # Shape: (batch, seq, 2)
    
    # Sum across the 2 dimensions (x, y) to get total squared error per timestep
    squared_error = masked_loss.sum(dim=2)  # Shape: (batch, seq)
    
    # Take square root to get Euclidean distance
    distances = torch.sqrt(squared_error)  # Shape: (batch, seq)
    
    # Sum distances per trajectory
    total_distances = torch.sum(distances, dim=1)  # Shape: (batch,)
    
    # Count valid points per trajectory (divide by 2 because each point has x,y)
    num_valid_points_per_traj = torch.sum(mask, dim=(1, 2)) / 2  # Shape: (batch,)
    
    # Handle division by zero
    valid_mask = (num_valid_points_per_traj > 0)
    if not valid_mask.any().item():
        return torch.tensor(0.0, device=y.device)
    
    # Calculate ADE for valid trajectories
    ade_per_traj = torch.zeros_like(total_distances)
    ade_per_traj[valid_mask] = total_distances[valid_mask] / num_valid_points_per_traj[valid_mask]
    
    # Calculate accuracy
    valid_ade = ade_per_traj[valid_mask]
    accuracy = torch.mean((valid_ade < threshold).float())
    return accuracy


def compute_all_metrics(y, y_pred):
    """Compute trajectory prediction metrics (ADE, FDE, and accuracy)"""
    metrics = {}
    try:
        ade = average_displacement_error(y, y_pred)
        fde = final_displacement_error(y, y_pred)
        acc_2px = trajectory_accuracy(y, y_pred, threshold=2.0)
        acc_5px = trajectory_accuracy(y, y_pred, threshold=5.0)
        
        # Ensure all results are scalar values
        metrics['ADE'] = float(ade)
        metrics['FDE'] = float(fde)
        metrics['Accuracy_2px'] = float(acc_2px)
        metrics['Accuracy_5px'] = float(acc_5px)
        
    except Exception as e:
        print(f"Error computing metrics: {e}")
        print(f"y shape: {y.shape}, y_pred shape: {y_pred.shape}")
        # Fallback to basic metrics
        metrics['ADE'] = 0.0
        metrics['FDE'] = 0.0
        metrics['Accuracy_2px'] = 0.0
        metrics['Accuracy_5px'] = 0.0
    return metrics
