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
            print(f"Loaded data shape: {df.shape}")

            # Convert the first column to datetime and set as index
            df['Date'] = pd.to_datetime(df.iloc[:, 0])
            df.set_index('Date', inplace=True)

            # Drop rows where all values are NaN
            df = df.dropna(how='all')

            # Print column info for debugging
            print("\nColumn info before conversion:")
            for col in df.columns:
                print(f"{col}: {df[col].dtype}, NaN count: {df[col].isna().sum()}")

            # Convert all numeric columns from European format (comma) to standard format (dot)
            numeric_columns = df.select_dtypes(include=['object']).columns
            for col in numeric_columns:
                try:
                    # Replace comma with dot and convert to float
                    df[col] = df[col].str.replace(',', '.').astype(float)
                except (AttributeError, ValueError):
                    try:
                        # Try direct conversion for non-string numeric values
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except Exception as e:
                        print(f"Warning: Could not convert column {col}: {str(e)}")

            # Forward fill missing values (appropriate for time series data)
            df = df.ffill()

            # Backward fill any remaining missing values at the start
            df = df.bfill()

            print("\nColumn info after conversion:")
            for col in df.columns:
                print(f"{col}: {df[col].dtype}, NaN count: {df[col].isna().sum()}")

            return df

        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")

    @staticmethod
    def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target variables."""
        try:
            # Print data info for debugging
            print("\nPreparing data:")
            print(f"Total columns: {len(df.columns)}")
            print(f"Numeric columns: {len(df.select_dtypes(include=[np.number]).columns)}")

            # Ensure 'Close' column exists
            if 'Close' not in df.columns:
                raise ValueError("Required 'Close' column not found in dataset")

            # Separate features and target
            y = df['Close'].values
            numeric_df = df.select_dtypes(include=[np.number])
            X = numeric_df.drop('Close', axis=1, errors='ignore').values

            # Validate data
            invalid_X = np.isnan(X).sum()
            invalid_y = np.isnan(y).sum()

            if invalid_X > 0 or invalid_y > 0:
                print(f"\nInvalid values found:")
                print(f"Features (X): {invalid_X} invalid values")
                print(f"Target (y): {invalid_y} invalid values")
                raise ValueError(f"Dataset contains missing or invalid values: {invalid_X} in features, {invalid_y} in target")

            print(f"\nFinal data shape:")
            print(f"X shape: {X.shape}")
            print(f"y shape: {y.shape}")

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