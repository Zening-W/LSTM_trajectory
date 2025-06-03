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
from utils import average_displacement_error

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


def train():

    if not os.path.exists(path_to_save_result):
        os.makedirs(path_to_save_result)
    
    # Prepare dataloader
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


def test():
    model_path = os.path.join(file_dir, 'runs', 'run_20241204051634', 'best_model_epoch88.pth')
    test_file = "2024-07-07_2024-07-07_17-00-03"
    visualize(model_path, test_file)


if __name__ == '__main__':
    train()
    # test()

            


    

    