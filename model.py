import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List

class LSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions

class GoldPricePredictor:
    def __init__(self):
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
    def prepare_sequences(self, X: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i + seq_length])
            y_seq.append(y[i + seq_length])
        return torch.FloatTensor(X_seq), torch.FloatTensor(y_seq)
    
    def train(self, X: np.ndarray, y: np.ndarray, params: dict) -> List[float]:
        # Scale data
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1))
        
        # Prepare sequences
        X_seq, y_seq = self.prepare_sequences(X_scaled, y_scaled, params['seq_length'])
        
        # Initialize model
        self.model = LSTMModel(
            input_dim=X.shape[1],
            hidden_dim=params['hidden_dim'],
            num_layers=params['num_layers'],
            dropout=params['dropout']
        )
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=params['learning_rate'])
        
        losses = []
        self.model.train()
        
        for epoch in range(params['epochs']):
            optimizer.zero_grad()
            outputs = self.model(X_seq)
            loss = criterion(outputs, y_seq)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        return losses
    
    def predict(self, X: np.ndarray, seq_length: int) -> np.ndarray:
        self.model.eval()
        X_scaled = self.scaler_X.transform(X)
        X_seq = torch.FloatTensor(X_scaled[-seq_length:]).unsqueeze(0)
        
        with torch.no_grad():
            prediction = self.model(X_seq)
            prediction = self.scaler_y.inverse_transform(prediction.numpy())
            
        return prediction.flatten()
