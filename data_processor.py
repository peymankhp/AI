import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict

class DataProcessor:
    @staticmethod
    def load_data(file) -> pd.DataFrame:
        """Load and preprocess the Excel data, handling European number format."""
        try:
            # Read Excel file
            df = pd.read_excel(file)

            # Convert the first column to datetime and set as index
            df['Date'] = pd.to_datetime(df.iloc[:, 0])
            df.set_index('Date', inplace=True)

            # Convert all numeric columns from European format (comma) to standard format (dot)
            numeric_columns = df.select_dtypes(include=['object']).columns
            for col in numeric_columns:
                try:
                    # Replace comma with dot and convert to float
                    df[col] = df[col].str.replace(',', '.').astype(float)
                except (AttributeError, ValueError):
                    # If conversion fails, try direct conversion
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except Exception as e:
                        print(f"Warning: Could not convert column {col}: {str(e)}")

            return df

        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")

    @staticmethod
    def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target variables."""
        try:
            # Ensure 'Close' column exists
            if 'Close' not in df.columns:
                raise ValueError("Required 'Close' column not found in dataset")

            # Separate features and target
            y = df['Close'].values
            X = df.drop('Close', axis=1).select_dtypes(include=[np.number]).values

            # Check for invalid values
            if np.isnan(X).any() or np.isnan(y).any():
                raise ValueError("Dataset contains missing or invalid values")

            return X, y

        except Exception as e:
            raise ValueError(f"Error preparing data: {str(e)}")

    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred)
        }

        return metrics

    @staticmethod
    def resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
        """Resample data to weekly frequency."""
        return df.resample('W').mean()