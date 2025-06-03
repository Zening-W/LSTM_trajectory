import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EncoderLSTM(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, share_embedding):
        super().__init__()
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)
        self.share_embedding = share_embedding
    
    def forward(self, x):
        x_embedding = self.share_embedding(x)
        output, (h_n, c_n) = self.lstm(x_embedding)
        return h_n, c_n


class DecoderLSTM(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, share_embedding):
        super().__init__()
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)
        self.share_embedding = share_embedding # Input layer 
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        ) # Output layer
    
    def forward(self, x_0, h_n, c_n, forecast_window):

        h_t = h_n
        c_t = c_n
        x_t = x_0[:, torch.newaxis, :] # (N, 1, F)
        output = torch.zeros(x_0.shape[0], forecast_window, x_0.shape[1], device=x_0.device) # (N, T, F)

        for t in range(forecast_window):
            # Input features to embeddings
            x_t = self.share_embedding(x_t) # (N, 1, D)
            # Embedding to hidden state
            x_t, (h_t, c_t) = self.lstm(x_t, (h_t, c_t)) # (N, 1, H)
            # Hidden state to output features
            x_t = self.fc(x_t) # (N, 1, F)
            output[:, t, :] = x_t.squeeze(1)
        
        return output


class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, forecast_window, num_layers=2):
        super().__init__()
        self.share_embedding = nn.Linear(input_size, embedding_size)
        self.encoder = EncoderLSTM(embedding_size, hidden_size, num_layers, self.share_embedding)
        self.decoder = DecoderLSTM(input_size, embedding_size, hidden_size, num_layers, self.share_embedding)
        self.forecast_window = forecast_window
    
    def forward(self, x):
        """
        Compute forecasting results based on inputs.

        Inputs:
        - x: Input trajectory timesteps, of shape (N, T, F)

        Returns:
        - output: Forecasted future trajectory timesteps, of shape (N, T, F)  
        """

        x_encoder = x[:, :-1, :]
        x_0 = x[:, -1, :]
        h_n, c_n = self.encoder(x_encoder)
        output = self.decoder(x_0, h_n, c_n, self.forecast_window)
        return output


if __name__ == '__main__':

    input_size = 2
    embedding_size = 32
    hidden_size = 256
    forecast_window = 12
    seq_encoder = 12
    batch_size = 64
    num_layers = 4

    model = Seq2SeqLSTM(input_size, embedding_size, hidden_size, forecast_window, num_layers)

    x = torch.randn(batch_size, seq_encoder, input_size)

    output = model(x)

    print(output.shape)
