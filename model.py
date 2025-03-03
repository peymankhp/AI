import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List
import logging

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
        """Prepare sequences for LSTM, ensuring X and y have matching dimensions."""
        if len(X) != len(y):
            raise ValueError(f"Input X and y must have same length. Got X: {len(X)}, y: {len(y)}")

        if seq_length >= len(X):
            raise ValueError(f"Sequence length {seq_length} must be less than data length {len(X)}")

        X_seq, y_seq = [], []
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:(i + seq_length)])
            y_seq.append(y[i + seq_length])

        print(f"Created sequences - X shape: {len(X_seq)}x{seq_length}x{X.shape[1]}, y shape: {len(y_seq)}")
        return torch.FloatTensor(np.array(X_seq)), torch.FloatTensor(np.array(y_seq))

    def train(self, X: np.ndarray, y: np.ndarray, params: dict) -> List[float]:
        """Train the LSTM model with proper sequence preparation."""
        print(f"Initial data shapes - X: {X.shape}, y: {y.shape}")

        # Scale data
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1))

        print(f"Scaled data shapes - X: {X_scaled.shape}, y: {y_scaled.shape}")

        # Prepare sequences
        X_seq, y_seq = self.prepare_sequences(X_scaled, y_scaled, params['seq_length'])

        print(f"Training sequences - X: {X_seq.shape}, y: {y_seq.shape}")

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

            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{params['epochs']}, Loss: {loss.item():.6f}")

        return losses

    def predict(self, X: np.ndarray, seq_length: int) -> np.ndarray:
        """Generate predictions with proper sequence handling."""
        if not self.model:
            raise ValueError("Model must be trained before making predictions")

        self.model.eval()
        X_scaled = self.scaler_X.transform(X)

        # Prepare sequences for prediction
        predictions = []
        for i in range(len(X_scaled) - seq_length):
            X_seq = torch.FloatTensor(X_scaled[i:i + seq_length]).unsqueeze(0)
            with torch.no_grad():
                pred = self.model(X_seq)
                pred = self.scaler_y.inverse_transform(pred.numpy())
                predictions.append(pred[0][0])

        # Pad the beginning with NaN values to match input length
        padding = [np.nan] * seq_length
        predictions = padding + predictions

        return np.array(predictions)