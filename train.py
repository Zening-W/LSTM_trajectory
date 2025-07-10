import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch import optim
import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import tqdm

from model import Seq2SeqLSTM
from dataloader import TrajDataset
from utils import average_displacement_error, compute_all_metrics

# Public variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

cur_time = int(datetime.now().strftime('%Y%m%d%H%M%S'))

file_dir = os.path.dirname(os.path.abspath(__file__))
path_to_save_result = os.path.join(file_dir, 'runs', 'run_' + str(cur_time))
path_to_save_test = os.path.join(file_dir, 'test', 'test_' + str(cur_time))


img_width = 720 
img_height = 480  
fps = 10

# Public variables of model hyperparameters
input_size = 2  # input feature (x, y)
input_len = 12
output_len = 12
embedding_size = 32
hidden_size = 256
forecast_window = output_len
num_layers = 2

# Public variables of training parameters
learning_rate = 1e-3
batch_size = 16
num_epochs = 100


def train(test_files_to_exclude=None):
    """
    Train the model with option to exclude specific files from training
    to prevent data leakage when those files are used for testing
    """
    if not os.path.exists(path_to_save_result):
        os.makedirs(path_to_save_result)
    
    # Prepare dataloader - exclude test files if specified
    if test_files_to_exclude:
        # Create training dataset excluding test files
        all_files = ["2024-07-07_2024-07-07_17-00-03.txt", 
                     "2024-07-07_2024-07-07_17-15-00.txt", 
                     "2024-07-07_2024-07-07_17-30-00.txt", 
                     "2024-07-07_2024-07-07_17-45-00.txt"]
        training_files = [f for f in all_files if f not in test_files_to_exclude]
        print(f"Training with files: {training_files}")
        print(f"Excluding test files: {test_files_to_exclude}")
        dataset = TrajDataset(list_filename=training_files)
    else:
        dataset = TrajDataset()
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False)

    # Initialize model, optimizer, loss function and so on
    model = Seq2SeqLSTM(input_size, embedding_size, hidden_size, forecast_window, num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), learning_rate, weight_decay=1e-4)
    mse_loss = nn.MSELoss(reduction="none")
    list_loss = []
    best_ade = float('inf')
    
    # Create training log file
    training_log_path = os.path.join(path_to_save_result, 'training.txt')
    with open(training_log_path, 'w') as log_file:
        log_file.write("Training Log\n")
        log_file.write("=" * 50 + "\n")

    for epoch in range(num_epochs):
        loss_epoch = 0
        model.train()
        for x, y, frame_idx in train_dataloader:
            x = x.to(device)
            y = y.to(device)
            
            mask = (y != -1).to(device)  # (N, T, 2)
            y_pred = model(x)  # (N, T, 2)
            loss = torch.sum(mse_loss(y_pred, y) * mask) / torch.sum(mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
            list_loss.append(loss.item())
        loss_epoch /= len(train_dataloader)
        print(f'Epoch {epoch} loss: {loss_epoch}')
    
        model.eval()
        with torch.no_grad():
            ade_epoch = 0
            for x_val, y_val, frame_idx in val_dataloader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                y_val_pred = model(x_val)
                ade = average_displacement_error(y_val, y_val_pred)
                ade_epoch += ade
            ade_epoch /= len(val_dataloader)
            print(f'Epoch {epoch} ade: {ade_epoch}')
            
            # Log to file
            with open(training_log_path, 'a') as log_file:
                log_file.write(f"Epoch {epoch} loss: {loss_epoch}\n")
                log_file.write(f"Epoch {epoch} ade: {ade_epoch}\n")
            
            if ade_epoch < best_ade:
                torch.save(model.state_dict(), os.path.join(path_to_save_result, f'best_model_epoch{epoch}.pth'))
                best_ade = ade_epoch
    

    plt.plot(range(len(list_loss)), list_loss, label="training loss")
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(path_to_save_result, 'training_loss.png')) 

    return model


def visualize(model_path, test_file):
    
    if not os.path.exists(path_to_save_test):
        os.makedirs(path_to_save_test)

    line_thickness = 1
    circle_thickness = 3

    path_to_read_frames = os.path.join(file_dir, 'data', 'frames', test_file)

    test_dataset = TrajDataset(list_filename=[test_file+'.txt'])
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = Seq2SeqLSTM(input_size, embedding_size, hidden_size, forecast_window, num_layers)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    with torch.no_grad():
        # img_no = 1
        for x, y, frame_idx in test_dataloader:
            y_pred = model(x)
            x, y, y_pred, frame_idx = x.squeeze(0).numpy(), y.squeeze(0).numpy(), y_pred.squeeze(0).numpy(), frame_idx.squeeze(0)
            mask = (y != -1)
            y_masked = y * mask
            y_pred_masked = y_pred * mask
            # image = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
            image_path = os.path.join(path_to_save_test, f'traj_{frame_idx}.png')
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
            else:
                image = cv2.imread(os.path.join(path_to_read_frames, f'{frame_idx}.png'))

            for i in range(len(x) - 1):
                start_point = (int(x[i][0]), int(x[i][1]))
                end_point = (int(x[i + 1][0]), int(x[i + 1][1]))
                color = (255, 255, 0)   
                cv2.line(image, start_point, end_point, color, line_thickness)
                cv2.circle(image, end_point, circle_thickness, color, -1)
                if i == 0:
                    cv2.circle(image, start_point, circle_thickness, color, -1)
            for i in range(len(y_masked) - 1):
                start_point = (int(y_masked[i][0]), int(y_masked[i][1]))
                end_point =  (int(y_masked[i + 1][0]), int(y_masked[i + 1][1]))
                color = (0, 255, 0)  
                if i == 0:
                    cv2.circle(image, start_point, circle_thickness, color, -1)
                if end_point == (0, 0):
                    break
                cv2.line(image, start_point, end_point, color, line_thickness)
                cv2.circle(image, end_point, circle_thickness, color, -1)
            for i in range(len(y_pred_masked) - 1):
                start_point = (int(y_pred_masked[i][0]), int(y_pred_masked[i][1]))
                end_point = (int(y_pred_masked[i + 1][0]), int(y_pred_masked[i + 1][1]))
                color = (255, 0, 0) 
                if i == 0:
                    cv2.circle(image, start_point, circle_thickness, color, -1)
                if end_point == (0, 0):
                    break
                cv2.line(image, start_point, end_point, color, line_thickness)
                cv2.circle(image, end_point, circle_thickness, color, -1)
            
            cv2.imwrite(os.path.join(path_to_save_test, f'traj_{frame_idx}.png'), image)
            # img_no += 1


def evaluate_model(model_path, test_file):
    """
    Comprehensive evaluation function for edge devices
    Returns detailed metrics for trajectory prediction performance
    """
    print(f"Evaluating model: {model_path}")
    print(f"Test file: {test_file}")
    
    # Load test data
    test_dataset = TrajDataset(list_filename=[test_file+'.txt'])
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Load model
    model = Seq2SeqLSTM(input_size, embedding_size, hidden_size, forecast_window, num_layers)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    # Evaluation metrics
    all_metrics = []
    
    with torch.no_grad():
        for x, y, frame_idx in test_dataloader:
            y_pred = model(x)
            
            # Compute metrics for this sample
            metrics = compute_all_metrics(y, y_pred)
            all_metrics.append(metrics)
            
            print(f"Frame {frame_idx.item()}: ADE={metrics['ADE']:.2f}, FDE={metrics['FDE']:.2f}, "
                  f"Acc_2px={metrics['Accuracy_2px']:.3f}, Acc_5px={metrics['Accuracy_5px']:.3f}")
    
    # Aggregate results
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
    
    print("\n=== FINAL EVALUATION RESULTS ===")
    print(f"Average ADE: {avg_metrics['ADE']:.2f} pixels")
    print(f"Average FDE: {avg_metrics['FDE']:.2f} pixels")
    print(f"Accuracy (2px threshold): {avg_metrics['Accuracy_2px']:.3f}")
    print(f"Accuracy (5px threshold): {avg_metrics['Accuracy_5px']:.3f}")
    print("================================")
    
    # Save evaluation results to file
    # Get the run directory from model_path
    run_dir = os.path.dirname(model_path)
    evaluation_log_path = os.path.join(run_dir, 'evaluation.txt')
    
    with open(evaluation_log_path, 'w') as log_file:
        log_file.write("Evaluation Results\n")
        log_file.write("=" * 50 + "\n")
        log_file.write(f"Model: {os.path.basename(model_path)}\n")
        log_file.write(f"Test File: {test_file}\n")
        log_file.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        log_file.write("Frame-by-Frame Results:\n")
        log_file.write("-" * 30 + "\n")
        for i, metrics in enumerate(all_metrics):
            log_file.write(f"Frame {i}: ADE={metrics['ADE']:.2f}, FDE={metrics['FDE']:.2f}, "
                          f"Acc_2px={metrics['Accuracy_2px']:.3f}, Acc_5px={metrics['Accuracy_5px']:.3f}\n")
        
        log_file.write("\nFinal Aggregated Results:\n")
        log_file.write("-" * 30 + "\n")
        log_file.write(f"Average ADE: {avg_metrics['ADE']:.2f} pixels\n")
        log_file.write(f"Average FDE: {avg_metrics['FDE']:.2f} pixels\n")
        log_file.write(f"Accuracy (2px threshold): {avg_metrics['Accuracy_2px']:.3f}\n")
        log_file.write(f"Accuracy (5px threshold): {avg_metrics['Accuracy_5px']:.3f}\n")
    
    print(f"\nEvaluation results saved to: {evaluation_log_path}")
    
    return avg_metrics


def test():
    # Use the same logic as evaluate() to find the latest model
    runs_dir = os.path.join(file_dir, 'runs')
    run_dirs = [d for d in os.listdir(runs_dir) if d.startswith('run_')]
    run_dirs.sort()
    latest_run = run_dirs[-1] if run_dirs else 'run_20250707010718'  # fallback
    
    # Find the model with the highest epoch number
    run_path = os.path.join(file_dir, 'runs', latest_run)
    model_files = [f for f in os.listdir(run_path) if f.startswith('best_model_epoch') and f.endswith('.pth')]
    
    if not model_files:
        print(f"No model files found in {latest_run}")
        return
    
    # Extract epoch numbers and find the highest
    epoch_numbers = []
    for model_file in model_files:
        try:
            epoch_num = int(model_file.replace('best_model_epoch', '').replace('.pth', ''))
            epoch_numbers.append(epoch_num)
        except ValueError:
            continue
    
    if not epoch_numbers:
        print(f"No valid model files found in {latest_run}")
        return
    
    highest_epoch = max(epoch_numbers)
    model_path = os.path.join(file_dir, 'runs', latest_run, f'best_model_epoch{highest_epoch}.pth')
    
    # Use the same test file as evaluation (the excluded file)
    test_file = "2024-07-07_2024-07-07_17-15-00"
    
    print(f"Visualizing model from: {latest_run}")
    print(f"Using epoch: {highest_epoch}")
    print(f"Visualizing test file: {test_file}")
    
    visualize(model_path, test_file)

def evaluate():
    # Use the newly trained model (will be in the latest run directory)
    # Get the most recent run directory
    runs_dir = os.path.join(file_dir, 'runs')
    run_dirs = [d for d in os.listdir(runs_dir) if d.startswith('run_')]
    run_dirs.sort()
    latest_run = run_dirs[-1] if run_dirs else 'run_20250707010718'  # fallback
    
    # Find the model with the highest epoch number
    run_path = os.path.join(file_dir, 'runs', latest_run)
    model_files = [f for f in os.listdir(run_path) if f.startswith('best_model_epoch') and f.endswith('.pth')]
    
    if not model_files:
        print(f"No model files found in {latest_run}")
        return
    
    # Extract epoch numbers and find the highest
    epoch_numbers = []
    for model_file in model_files:
        try:
            epoch_num = int(model_file.replace('best_model_epoch', '').replace('.pth', ''))
            epoch_numbers.append(epoch_num)
        except ValueError:
            continue
    
    if not epoch_numbers:
        print(f"No valid model files found in {latest_run}")
        return
    
    highest_epoch = max(epoch_numbers)
    model_path = os.path.join(file_dir, 'runs', latest_run, f'best_model_epoch{highest_epoch}.pth')
    
    # Test on the excluded file
    test_file = "2024-07-07_2024-07-07_17-15-00"
    
    print(f"Using model from: {latest_run}")
    print(f"Using epoch: {highest_epoch}")
    print(f"Testing on: {test_file}")
    
    try:
        evaluate_model(model_path, test_file)
        print(f"Successfully evaluated with {test_file}")
    except Exception as e:
        print(f"Failed with {test_file}: {e}")
        # Fallback to other files if needed
        fallback_files = [
            "2024-07-07_2024-07-07_17-30-00",
            "2024-07-07_2024-07-07_17-45-00",
            "2024-07-07_2024-07-07_17-00-03"
        ]
        
        for fallback_file in fallback_files:
            print(f"\nTrying fallback file: {fallback_file}")
            try:
                evaluate_model(model_path, fallback_file)
                print(f"Successfully evaluated with {fallback_file}")
                break
            except Exception as e2:
                print(f"Failed with {fallback_file}: {e2}")
                continue

if __name__ == '__main__':
    # Train new model excluding 17-15-00 as test file
    #train(test_files_to_exclude=["2024-07-07_2024-07-07_17-15-00.txt"])
    
    # Evaluate on the excluded test file
    evaluate()
    
    #test()

            


    

    