# -*- coding: utf-8 -*-
"""Professional Gold Price Prediction System"""
# System imports
import pandas as pd  # Data manipulation and analysis
import numpy as np   # Numerical computations
from sklearn.preprocessing import MinMaxScaler  # Data normalization
from sklearn.model_selection import TimeSeriesSplit  # Time-series validation
import logging  # For logging
import os  # For file operations
from datetime import datetime  # For timestamps
import tensorflow as tf  # Explicitly import tensorflow
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For better plot styling

# Set plot style
plt.style.use('seaborn-v0_8')  # Use a built-in style
sns.set_palette("husl")

# Set up base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, 'gold_price_prediction.log')),
        logging.StreamHandler()
    ]
)

# Configure TensorFlow logging
tf.get_logger().setLevel(logging.ERROR)  # Reduce TensorFlow logging verbosity

# Create model directory if it doesn't exist
MODEL_DIR = os.path.join(BASE_DIR, 'models')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Data file paths
DATA_FILES = {
    'train': os.path.join(BASE_DIR, 'Gold_Price_TA_training.xlsx'),
    'test': os.path.join(BASE_DIR, 'Gold_Price_TA_test.xlsx')
}

# Verify data files exist
for name, path in DATA_FILES.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required data file not found: {path}")

# ==============================================
# 1. DATA LOADING AND PREPROCESSING
# ==============================================

def resample_to_weekly(df):
    """
    Resample daily data to weekly frequency
    Args:
        df: Daily DataFrame with DateTimeIndex
    Returns:
        Weekly resampled DataFrame
    """
    # Resample using last observation of the week for all columns
    weekly_df = df.resample('W').last()
    logging.info(f"Resampled data from {len(df)} daily to {len(weekly_df)} weekly observations")
    return weekly_df

def load_and_preprocess_data(file_path, columns_to_keep=None):
    """
    Comprehensive data loading and cleaning pipeline
    Args:
        file_path: Path to Excel data file
        columns_to_keep: List of columns to keep (if None, keep all columns)
    Returns:
        Tuple of (daily_df, weekly_df)
    """
    try:
        logging.info(f"Loading data from {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} was not found")

        # Read Excel file with optimized settings
        df = pd.read_excel(
            file_path,
            decimal=',',  # European number format
            engine='openpyxl',  # More memory efficient Excel engine
            dtype={
                'Date': str,  # Read dates as strings initially
            }
        )
        
        if df.empty:
            raise ValueError("The loaded file is empty")
            
        # Log initial data info
        logging.info(f"Initial data shape: {df.shape}")
        
        # Try different date formats
        date_formats = ['%d.%m.%y', '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y']
        date_converted = False
        
        for date_format in date_formats:
            try:
                df['Date'] = pd.to_datetime(df['Date'], format=date_format, errors='raise')
                date_converted = True
                logging.info(f"Successfully parsed dates using format: {date_format}")
                break
            except ValueError:
                continue
        
        if not date_converted:
            try:
                df['Date'] = pd.to_datetime(df['Date'], errors='raise')
                date_converted = True
                logging.info("Successfully parsed dates using automatic format detection")
            except ValueError as e:
                logging.error(f"Failed to parse dates: {str(e)}")
                sample_dates = df['Date'].head()
                logging.error(f"Sample of unparseable dates: {sample_dates.tolist()}")
                raise ValueError("Could not parse dates in any recognized format")
        
        # Keep only specified columns if provided (before setting index)
        if columns_to_keep is not None:
            # Add 'Date' to columns_to_keep temporarily
            columns_with_date = ['Date'] + columns_to_keep
            df = df[columns_with_date]
            logging.info(f"Keeping {len(columns_to_keep)} common columns")
        
        # Set Date as index and sort chronologically
        df = df.sort_values('Date').set_index('Date')
        
        # Convert European number format and handle non-numeric values
        numeric_conversion_errors = []
        for col in df.select_dtypes(include=[object]).columns:
            logging.info(f"Converting column {col} to numeric")
            df[col] = df[col].str.replace(',', '.', regex=False)
            before_conversion = df[col].copy()
            df[col] = pd.to_numeric(df[col], errors='coerce')
            num_errors = df[col].isnull().sum() - before_conversion.isnull().sum()
            if num_errors > 0:
                numeric_conversion_errors.append(f"{col}: {num_errors} conversion errors")
        
        if numeric_conversion_errors:
            logging.warning("Numeric conversion errors found:")
            for error in numeric_conversion_errors:
                logging.warning(error)
        
        # Handle missing values with sophisticated approach
        # 1. For time-series specific columns (identified by keywords)
        timeseries_cols = df.columns[df.columns.str.contains('|'.join(['Price', 'Index', 'Rate', 'GDP', 'CPI']), case=False)]
        if not timeseries_cols.empty:
            df[timeseries_cols] = df[timeseries_cols].interpolate(method='time', limit_direction='both', limit=10)
        
        # 2. For other numerical columns, use forward fill with a 5-day window
        df = df.ffill(limit=5)
        
        # 3. Use backward fill with a 5-day window
        df = df.bfill(limit=5)
        
        # 4. For any remaining gaps, use 30-day rolling mean
        df = df.fillna(df.rolling(window=30, min_periods=1, center=True).mean())
        
        # 5. If still have missing values, use global mean for that column
        if df.isnull().sum().any():
            df = df.fillna(df.mean())
        
        # Final check for missing values
        missing_after = df.isnull().sum()
        if missing_after.any():
            logging.error("Still have missing values after cleaning:")
            for col in missing_after[missing_after > 0].index:
                logging.error(f"{col}: {missing_after[col]} missing values")
                missing_rows = df[df[col].isnull()].head()
                logging.error(f"Sample rows with missing values in {col}:")
                logging.error(missing_rows)
            raise ValueError("Unable to handle all missing values in the dataset")
        
        # Create weekly version
        weekly_df = resample_to_weekly(df)
        
        logging.info(f"Successfully loaded and preprocessed data. Daily shape: {df.shape}, Weekly shape: {weekly_df.shape}")
        return df, weekly_df
        
    except Exception as e:
        logging.error(f"Error processing the data: {str(e)}")
        raise

def calculate_prediction_accuracy(y_true, y_pred):
    """
    Calculate various accuracy metrics for predictions
    Args:
        y_true: True values
        y_pred: Predicted values
    Returns:
        Dictionary of accuracy metrics
    """
    # Calculate percentage errors
    percentage_errors = np.abs((y_true - y_pred) / y_true) * 100
    
    # Calculate directional accuracy
    direction_true = np.diff(y_true) > 0
    direction_pred = np.diff(y_pred) > 0
    directional_accuracy = np.mean(direction_true == direction_pred) * 100
    
    metrics = {
        'mape': np.mean(percentage_errors),  # Mean Absolute Percentage Error
        'median_pe': np.median(percentage_errors),  # Median Percentage Error
        'rmse': np.sqrt(np.mean((y_true - y_pred) ** 2)),  # Root Mean Square Error
        'mae': np.mean(np.abs(y_true - y_pred)),  # Mean Absolute Error
        'r2': 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)),  # R-squared
        'directional_accuracy': directional_accuracy  # Directional Accuracy
    }
    
    return metrics

def get_common_columns(dataframes, missing_threshold=0.3):
    """
    Identify common columns across all dataframes with acceptable missing value ratios
    Args:
        dataframes: List of pandas DataFrames to analyze
        missing_threshold: Maximum acceptable ratio of missing values (default: 0.3 or 30%)
    Returns:
        List of column names that meet the criteria
    """
    try:
        # Get set of columns from first dataframe
        common_cols = set(dataframes[0].columns)
        
        # Find intersection with all other dataframes
        for df in dataframes[1:]:
            common_cols = common_cols.intersection(df.columns)
        
        # Convert to list and sort
        common_cols = sorted(list(common_cols))
        logging.info(f"Found {len(common_cols)} columns common to all datasets")
        
        # Check missing value ratios
        valid_columns = []
        for col in common_cols:
            # Skip Date column as it's handled separately
            if col == 'Date':
                continue
                
            # Check missing ratio in all dataframes
            max_missing_ratio = max(
                df[col].isnull().mean() 
                for df in dataframes
            )
            
            if max_missing_ratio <= missing_threshold:
                valid_columns.append(col)
            else:
                logging.warning(f"Column {col} excluded due to high missing ratio: {max_missing_ratio:.2%}")
        
        logging.info(f"Retained {len(valid_columns)} columns after missing value filtering")
        return valid_columns
        
    except Exception as e:
        logging.error(f"Error in get_common_columns: {str(e)}")
        raise

# First load raw data to identify common columns
raw_train_df = pd.read_excel(DATA_FILES['train'])
raw_test_df = pd.read_excel(DATA_FILES['test'])

# Get common columns that meet missing value threshold
common_columns = get_common_columns([raw_train_df, raw_test_df])
logging.info(f"Identified {len(common_columns)} common columns across all datasets")

# Load and preprocess all datasets with common columns
train_daily_df, train_weekly_df = load_and_preprocess_data(DATA_FILES['train'], common_columns)
test_daily_df, test_weekly_df = load_and_preprocess_data(DATA_FILES['test'], common_columns)

# ==============================================
# 2. FEATURE ENGINEERING AND SCALING
# ==============================================

def scale_dataset(df, scaler, fit_scaler=False):
    """
    Scale the dataset using the provided scaler
    Args:
        df: DataFrame to scale
        scaler: sklearn scaler object
        fit_scaler: Whether to fit the scaler on this data
    Returns:
        Scaled numpy array
    """
    try:
        # Convert DataFrame to numpy array
        data = df.values
        
        # Fit scaler if requested
        if fit_scaler:
            data_scaled = scaler.fit_transform(data)
            logging.info(f"Fitted and transformed data with shape: {data_scaled.shape}")
        else:
            data_scaled = scaler.transform(data)
            logging.info(f"Transformed data with shape: {data_scaled.shape}")
        
        return data_scaled
        
    except Exception as e:
        logging.error(f"Error scaling dataset: {str(e)}")
        raise

# Define target variable (gold closing price)
target_column = 'Close'  # Value we want to predict

# Create feature list (all columns except target)
feature_columns = [col for col in common_columns if col != target_column]

# Initialize normalization scalers
daily_scaler = MinMaxScaler(feature_range=(0, 1))
weekly_scaler = MinMaxScaler(feature_range=(0, 1))

# Scale daily data
train_daily_scaled = scale_dataset(train_daily_df, daily_scaler, fit_scaler=True)
test_daily_scaled = scale_dataset(test_daily_df, daily_scaler)

# Scale weekly data
train_weekly_scaled = scale_dataset(train_weekly_df, weekly_scaler, fit_scaler=True)
test_weekly_scaled = scale_dataset(test_weekly_df, weekly_scaler)

# ==============================================
# 3. LSTM TIME-SERIES MODEL ARCHITECTURE
# ==============================================

from tensorflow.keras.models import Sequential  # Neural network container
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input  # Neural network layers
from tensorflow.keras.losses import Huber  # Huber loss function
from tensorflow.keras.optimizers import Adam  # Adam optimizer

def create_lstm_model(input_shape):
    """
    Build LSTM neural network for temporal pattern recognition
    Args:
        input_shape: (time_steps, features) format
    Returns:
        Compiled Keras model ready for training
    """
    try:
        logging.info(f"Creating LSTM model with input shape: {input_shape}")
        model = Sequential(name="GoldPriceLSTM")  # Initialize sequential model
        
        # Input layer
        model.add(Input(shape=input_shape, name="Input"))
        
        # First LSTM layer with return sequences for stacking
        model.add(LSTM(
            units=128,  # Increased memory units
            return_sequences=True,  # Pass full sequence to next layer
            name="LSTM1"
        ))
        model.add(Dropout(0.2))  # Add dropout for regularization
        
        # Second LSTM layer (returns single vector)
        model.add(LSTM(
            units=64,  # Increased dimensionality
            return_sequences=False,  # Final sequence output
            name="LSTM2"
        ))
        model.add(Dropout(0.2))  # Add dropout for regularization
        
        # Hidden dense layers for feature combination
        model.add(Dense(32, activation='relu', name="HiddenDense1"))
        model.add(Dropout(0.1))
        model.add(Dense(16, activation='relu', name="HiddenDense2"))
        
        # Output layer for price prediction (linear activation)
        model.add(Dense(1, name="Output"))  # Single value prediction
        
        # Compile model with adaptive learning rate and robust loss
        model.compile(
            optimizer=Adam(learning_rate=0.001),  # Configured Adam optimizer
            loss=Huber(delta=1.0),  # Huber loss with delta=1.0
            metrics=['mae', 'mse']  # Track both MAE and MSE
        )
        
        logging.info("LSTM model created successfully")
        model.summary(print_fn=logging.info)  # Log model architecture
        return model
        
    except Exception as e:
        logging.error(f"Error creating LSTM model: {str(e)}")
        raise

# ==============================================
# 4. RISK ANALYSIS COMPONENTS
# ==============================================

def calculate_value_at_risk(returns, confidence=0.95):
    """
    Calculate maximum expected loss using historical simulation
    Args:
        returns: Array of historical returns
        confidence: Probability threshold (95% default)
    Returns:
        VaR value (negative = potential loss)
    """
    return np.percentile(returns, 100 * (1 - confidence))  # Nth percentile loss

def monte_carlo_simulation(start_price, days, num_simulations, mu, sigma):
    """
    Generate stochastic price paths using Geometric Brownian Motion
    Args:
        start_price: Current market price
        days: Projection horizon
        num_simulations: Number of scenarios
        mu: Annualized return expectation
        sigma: Annualized volatility
    Returns:
        Matrix of simulated price paths (days x simulations)
    """
    # Calculate daily drift and volatility
    daily_return = mu / days  # Daily expected return
    daily_vol = sigma / np.sqrt(days)  # Daily volatility
    
    # Generate random daily returns
    shocks = np.random.normal(
        daily_return, 
        daily_vol, 
        (days, num_simulations)  # 2D array of returns
    )
    
    # Convert to price paths using cumulative product
    price_paths = start_price * np.cumprod(1 + shocks, axis=0)
    
    return price_paths

# ==============================================
# 5. MODEL TRAINING FRAMEWORK
# ==============================================

def train_time_series_model(model, X_train, y_train, validation_data, epochs=100):
    """
    Training process with early stopping and model saving
    Args:
        model: Untrained Keras model
        X_train: Training features
        y_train: Training labels
        validation_data: Tuple of (X_val, y_val)
        epochs: Maximum training iterations
    Returns:
        Training history metrics
    """
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    
    # Generate timestamp for model saving
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODEL_DIR, f'gold_price_model_{timestamp}.h5')
    
    # Configure callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',  # Watch validation loss
            patience=15,  # Increased patience
            restore_best_weights=True,  # Keep best model
            verbose=1
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(  # Add learning rate reduction
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    try:
        logging.info("Starting model training...")
        # Execute model training
        history = model.fit(
            X_train, y_train,
            validation_data=validation_data,  # Monitor generalization
            epochs=epochs,  # Maximum iterations
            batch_size=32,  # Samples per gradient update
            callbacks=callbacks,  # Activate callbacks
            verbose=1  # Show progress bar
        )
        
        logging.info(f"Model training completed. Best model saved to {model_path}")
        return history
        
    except Exception as e:
        logging.error(f"Error during model training: {str(e)}")
        raise

def create_sequences(data, window_size):
    """
    Create sequences for time series prediction
    Args:
        data: Scaled numpy array of shape (samples, features)
        window_size: Number of time steps to look back
    Returns:
        X: Input sequences of shape (samples, window_size, features)
        y: Target values of shape (samples, 1)
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size, 0])  # Assuming first column is target
    return np.array(X), np.array(y)

def plot_training_history(history):
    """
    Plot training history showing loss and metrics over epochs
    Args:
        history: Keras history object from model training
    """
    try:
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        
        # Plot training and validation loss
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss Over Epochs')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot MAE
        ax2.plot(history.history['mae'], label='Training MAE')
        ax2.plot(history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Mean Absolute Error Over Epochs')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)
        
        # Plot MSE
        ax3.plot(history.history['mse'], label='Training MSE')
        ax3.plot(history.history['val_mse'], label='Validation MSE')
        ax3.set_title('Mean Squared Error Over Epochs')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('MSE')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(BASE_DIR, 'training_history.png')
        plt.savefig(plot_path)
        plt.close()
        
        logging.info(f"Training history plot saved to {plot_path}")
        
    except Exception as e:
        logging.error(f"Error plotting training history: {str(e)}")
        raise

def plot_predictions(y_true, y_pred, title, save_path, dates=None):
    """
    Plot actual vs predicted values
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save the plot
        dates: Optional array of dates for x-axis
    """
    try:
        plt.figure(figsize=(15, 8))
        
        # Plot actual and predicted values
        if dates is not None:
            plt.plot(dates, y_true, label='Actual', alpha=0.8)
            plt.plot(dates, y_pred, label='Predicted', alpha=0.8)
            plt.xticks(rotation=45)
        else:
            plt.plot(y_true, label='Actual', alpha=0.8)
            plt.plot(y_pred, label='Predicted', alpha=0.8)
        
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Gold Price (USD)')
        plt.legend()
        plt.grid(True)
        
        # Add prediction accuracy metrics to the plot
        metrics = calculate_prediction_accuracy(y_true, y_pred)
        metrics_text = f"MAPE: {metrics['mape']:.2f}%\n"
        metrics_text += f"RMSE: ${metrics['rmse']:.2f}\n"
        metrics_text += f"Directional Accuracy: {metrics['directional_accuracy']:.2f}%"
        
        plt.text(0.02, 0.98, metrics_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(save_path)
        plt.close()
        
        logging.info(f"Prediction plot saved to {save_path}")
        
    except Exception as e:
        logging.error(f"Error plotting predictions: {str(e)}")
        raise

def evaluate_model(model, X_test, y_test, scaler, freq='daily'):
    """
    Evaluate model performance on test set
    Args:
        model: Trained Keras model
        X_test: Test features
        y_test: Test labels
        scaler: Scaler used for the data
        freq: Frequency of predictions ('daily' or 'weekly')
    Returns:
        Dictionary of evaluation metrics and predictions
    """
    try:
        logging.info(f"Evaluating {freq} model performance...")
        # Get model predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics on scaled data
        mse = np.mean((y_test - y_pred.flatten()) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - y_pred.flatten()))
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }
        
        # Inverse transform predictions and actual values
        n_features = X_test.shape[2]  # Number of features
        dummy_test = np.zeros((len(y_test), n_features))
        dummy_test[:, 0] = y_test  # First column is the target (Close price)
        y_test_actual = scaler.inverse_transform(dummy_test)[:, 0]
        
        dummy_pred = np.zeros((len(y_pred), n_features))
        dummy_pred[:, 0] = y_pred.flatten()
        y_pred_actual = scaler.inverse_transform(dummy_pred)[:, 0]
        
        # Calculate accuracy metrics
        accuracy_metrics = calculate_prediction_accuracy(y_test_actual, y_pred_actual)
        metrics.update(accuracy_metrics)
        
        # Plot predictions
        plot_path = os.path.join(BASE_DIR, f'predictions_{freq}.png')
        plot_predictions(
            y_test_actual,
            y_pred_actual,
            f'Gold Price Predictions vs Actual Values ({freq.capitalize()} Test Set)',
            plot_path
        )
        
        logging.info(f"{freq.capitalize()} model evaluation metrics:")
        for metric_name, metric_value in metrics.items():
            if metric_name in ['mape', 'directional_accuracy']:
                logging.info(f"{metric_name}: {metric_value:.2f}%")
            else:
                logging.info(f"{metric_name}: {metric_value:.4f}")
        
        return metrics, y_pred_actual
        
    except Exception as e:
        logging.error(f"Error during model evaluation: {str(e)}")
        raise

# ==============================================
# 6. MAIN EXECUTION FLOW
# ==============================================

if __name__ == "__main__":
    try:
        logging.info("Starting Gold Price Prediction System")
        
        # Log data shapes for debugging
        logging.info(f"Training data shape: {train_daily_df.shape}, {train_weekly_df.shape}")
        logging.info(f"Test data shape: {test_daily_df.shape}, {test_weekly_df.shape}")
        
        # Example risk analysis workflow
        price_returns = train_daily_df['Close'].pct_change().dropna()  # Daily returns
        var_95 = calculate_value_at_risk(price_returns)  # 95% confidence level
        logging.info(f"Daily 95% Value-at-Risk: {var_95*100:.2f}%")
        
        # Monte Carlo simulation example
        logging.info("Starting Monte Carlo simulation...")
        mc_simulations = monte_carlo_simulation(
            start_price=train_daily_df['Close'].iloc[-1],  # Latest price
            days=252,  # Trading days in year
            num_simulations=1000,  # Number of scenarios
            mu=price_returns.mean() * 252,  # Annualized return
            sigma=price_returns.std() * np.sqrt(252)  # Annualized volatility
        )
        logging.info("Monte Carlo simulation completed")
        
        # Create sequences for LSTM training
        window_size = 60
        logging.info(f"Creating sequences with window size: {window_size}")
        X_train_daily, y_train_daily = create_sequences(train_daily_scaled, window_size)
        X_train_weekly, y_train_weekly = create_sequences(train_weekly_scaled, window_size)
        X_test_daily, y_test_daily = create_sequences(test_daily_scaled, window_size)
        X_test_weekly, y_test_weekly = create_sequences(test_weekly_scaled, window_size)
        
        # Log sequence shapes
        logging.info(f"Training sequences shape: X_daily={X_train_daily.shape}, y_daily={y_train_daily.shape}, X_weekly={X_train_weekly.shape}, y_weekly={y_train_weekly.shape}")
        logging.info(f"Test sequences shape: X_daily={X_test_daily.shape}, y_daily={y_test_daily.shape}, X_weekly={X_test_weekly.shape}, y_weekly={y_test_weekly.shape}")
        
        # Create and train the LSTM model
        input_shape_daily = (window_size, train_daily_scaled.shape[1])
        input_shape_weekly = (window_size, train_weekly_scaled.shape[1])
        lstm_model_daily = create_lstm_model(input_shape_daily)
        lstm_model_weekly = create_lstm_model(input_shape_weekly)
        
        # Train the LSTM model
        history_daily = train_time_series_model(
            lstm_model_daily, 
            X_train_daily, 
            y_train_daily, 
            validation_data=(X_test_daily, y_test_daily),
            epochs=100
        )
        history_weekly = train_time_series_model(
            lstm_model_weekly, 
            X_train_weekly, 
            y_train_weekly, 
            validation_data=(X_test_weekly, y_test_weekly),
            epochs=100
        )
        
        # Plot training history
        plot_training_history(history_daily)
        plot_training_history(history_weekly)
        
        # Evaluate model performance and get predictions
        metrics_daily, y_pred_daily = evaluate_model(
            lstm_model_daily,
            X_test_daily,
            y_test_daily,
            daily_scaler,
            freq='daily'
        )
        metrics_weekly, y_pred_weekly = evaluate_model(
            lstm_model_weekly,
            X_test_weekly,
            y_test_weekly,
            weekly_scaler,
            freq='weekly'
        )
        
        # Compare daily vs weekly performance
        logging.info("\nModel Performance Comparison:")
        logging.info("============================")
        metrics_comparison = {
            'Daily MAPE': f"{metrics_daily['mape']:.2f}%",
            'Weekly MAPE': f"{metrics_weekly['mape']:.2f}%",
            'Daily Directional Accuracy': f"{metrics_daily['directional_accuracy']:.2f}%",
            'Weekly Directional Accuracy': f"{metrics_weekly['directional_accuracy']:.2f}%",
            'Daily RMSE': f"${metrics_daily['rmse']:.2f}",
            'Weekly RMSE': f"${metrics_weekly['rmse']:.2f}"
        }
        for metric, value in metrics_comparison.items():
            logging.info(f"{metric}: {value}")
        
        logging.info("Gold Price Prediction System completed successfully")
        
    except Exception as e:
        logging.error(f"An error occurred during execution: {str(e)}")
        raise