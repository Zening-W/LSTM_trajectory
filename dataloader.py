import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np


class TrajDataset(Dataset):
    def __init__(self, list_filename=["2024-07-07_2024-07-07_17-00-03.txt", 
                                      "2024-07-07_2024-07-07_17-15-00.txt", 
                                      "2024-07-07_2024-07-07_17-30-00.txt", 
                                      "2024-07-07_2024-07-07_17-45-00.txt"], 
                            input_len=12, 
                            output_len=12):
        
        self.data = []
        self.input_len = input_len
        self.output_len = output_len
        
        for filename in list_filename:
            grouped_trajectory = self._read_file(filename)

            for id, trajectory in grouped_trajectory: 
                trajectory = trajectory[['x', 'y', 'frame_idx']].values  # convert to numpy array of shape (len_traj, 3)
                num_features = 2  # features include x and y
                traj_len = trajectory.shape[0]
                num_samples = traj_len - input_len
                if num_samples > 0:
                    for i in range(num_samples):
                        input_seq = trajectory[i : i+input_len, :2]
                        if i+input_len+output_len < traj_len:
                            output_seq = trajectory[i+input_len : i+input_len+output_len, :2] 
                        else:
                            output_seq = trajectory[i+input_len : , :2]
                            output_seq = np.concatenate([output_seq, np.full((output_len - output_seq.shape[0], num_features), -1)], axis=0)

                        input_seq = torch.tensor(input_seq, dtype=torch.float32)
                        output_seq = torch.tensor(output_seq, dtype=torch.float32)

                        self.data.append((input_seq, output_seq, trajectory[i+input_len-1, 2]))  # (input_seq, output_seq, frame_idx)
    
    def _read_file(self, file_name):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_file = os.path.join(script_dir, "data", file_name)

        # 720 * 480  10fps
        df = pd.read_csv(csv_file, sep=' ', header=None, 
                         names=["frame_idx", "cls", "id", "bbox_left", "bbox_top", "bbox_w", "bbox_h"], # frame_idx + 1
                         usecols=range(7))

        df["x"] = df["bbox_left"] + df["bbox_w"] // 2
        df["y"] = df["bbox_top"] + df["bbox_h"] // 2
        df["w"] = df["bbox_w"]
        df["h"] = df["bbox_h"]

        df = df[["id", "frame_idx", "cls", "x", "y", "w", "h"]]

        grouped_trajectory = df.groupby("id")

        return grouped_trajectory
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]


if __name__ == '__main__':
    # list_filename=["2024-07-07_2024-07-07_17-00-03.txt",]
    dataset = TrajDataset()
    print(len(dataset))
    
        

    

