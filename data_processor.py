import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict

class DataProcessor:
    @staticmethod
    def load_data(file) -> pd.DataFrame:
        df = pd.read_excel(file)
        df['Date'] = pd.to_datetime(df.iloc[:, 0])
        df.set_index('Date', inplace=True)
        return df
    
    @staticmethod
    def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        # Separate features and target
        y = df['Close'].values
        X = df.drop('Close', axis=1).values
        
        return X, y
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred)
        }
        
        return metrics
    
    @staticmethod
    def resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
        return df.resample('W').mean()
