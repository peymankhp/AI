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

            # Convert numeric columns from European format (comma) to standard format (dot)
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

            # Drop columns with too many missing values (>30%)
            threshold = len(df) * 0.3
            columns_to_drop = df.columns[df.isna().sum() > threshold]
            if len(columns_to_drop) > 0:
                print(f"\nDropping columns with >30% missing values: {list(columns_to_drop)}")
                df = df.drop(columns=columns_to_drop)

            # Keep only relevant features (technical indicators and key commodities)
            relevant_columns = ['Close', 'Open', 'High', 'Low', 'Volume',
                              'Gold', 'Silver', 'Platinum',
                              'Total Index', 'Energy', 'Non-energy',
                              'Precious Metals', 'Base Metals',
                              'MACD Line', 'RSI Daily Gain', 'RSI Daily Loss']

            # Keep only columns that exist in the dataset
            existing_columns = [col for col in relevant_columns if col in df.columns]
            df = df[existing_columns]

            print("\nColumn info after cleaning:")
            for col in df.columns:
                print(f"{col}: {df[col].dtype}, NaN count: {df[col].isna().sum()}")

            if df.isna().any().any():
                problematic_cols = df.columns[df.isna().any()].tolist()
                raise ValueError(f"Still have missing values in columns: {problematic_cols}")

            return df

        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")

    @staticmethod
    def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target variables."""
        try:
            print("\nPreparing data:")
            print(f"Total columns: {len(df.columns)}")
            print(f"Available columns: {df.columns.tolist()}")

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
                print("\nDetailed feature validation:")
                feature_names = numeric_df.drop('Close', axis=1).columns
                for i, col in enumerate(feature_names):
                    nan_count = np.isnan(X[:, i]).sum()
                    if nan_count > 0:
                        print(f"Column '{col}' has {nan_count} invalid values")
                raise ValueError(f"Dataset contains invalid values: {invalid_X} in features, {invalid_y} in target")

            print(f"\nFinal data shape:")
            print(f"X shape: {X.shape}")
            print(f"y shape: {y.shape}")

            if len(X) != len(y):
                raise ValueError(f"Inconsistent number of samples: X has {len(X)}, y has {len(y)}")

            return X, y

        except Exception as e:
            raise ValueError(f"Error preparing data: {str(e)}")

    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        # Handle NaN values in predictions
        mask = ~np.isnan(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

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