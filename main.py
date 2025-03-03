import streamlit as st
import pandas as pd
import numpy as np
from data_processor import DataProcessor
from model import GoldPricePredictor
from visualizer import Visualizer
import io

# Page config
st.set_page_config(
    page_title="Gold Price Predictor",
    layout="wide"
)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = GoldPricePredictor()

# Title and description
st.title("Gold Price Prediction")
st.markdown("""
This application predicts gold prices using LSTM (Long Short-Term Memory) neural networks.
Upload your training and test datasets to begin.
""")

# File upload section
col1, col2 = st.columns(2)
with col1:
    training_file = st.file_uploader("Upload training data (Excel)", type=['xlsx'])
with col2:
    test_file = st.file_uploader("Upload test data (Excel)", type=['xlsx'])

if training_file and test_file:
    # Load and process data
    try:
        train_df = DataProcessor.load_data(training_file)
        test_df = DataProcessor.load_data(test_file)
        
        # Prepare data
        X_train, y_train = DataProcessor.prepare_data(train_df)
        X_test, y_test = DataProcessor.prepare_data(test_df)
        
        # Model parameters
        st.sidebar.header("Model Parameters")
        params = {
            'seq_length': st.sidebar.slider("Sequence Length", 5, 50, 10),
            'hidden_dim': st.sidebar.slider("Hidden Dimension", 32, 256, 64),
            'num_layers': st.sidebar.slider("Number of LSTM Layers", 1, 5, 2),
            'dropout': st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.2),
            'learning_rate': st.sidebar.select_slider(
                "Learning Rate",
                options=[0.0001, 0.001, 0.01, 0.1],
                value=0.001
            ),
            'epochs': st.sidebar.slider("Training Epochs", 10, 200, 50)
        }
        
        # Training section
        if st.button("Train Model"):
            with st.spinner("Training in progress..."):
                losses = st.session_state.predictor.train(X_train, y_train, params)
                
                # Plot training loss
                st.subheader("Training Loss")
                loss_fig = Visualizer.plot_training_loss(losses)
                st.plotly_chart(loss_fig, use_container_width=True)
                
                # Generate predictions
                train_pred = st.session_state.predictor.predict(X_train, params['seq_length'])
                test_pred = st.session_state.predictor.predict(X_test, params['seq_length'])
                
                # Calculate metrics
                train_metrics = DataProcessor.calculate_metrics(y_train[params['seq_length']:], train_pred)
                test_metrics = DataProcessor.calculate_metrics(y_test[params['seq_length']:], test_pred)
                
                # Display metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Training Metrics")
                    for metric, value in train_metrics.items():
                        st.metric(metric, f"{value:.4f}")
                
                with col2:
                    st.subheader("Test Metrics")
                    for metric, value in test_metrics.items():
                        st.metric(metric, f"{value:.4f}")
                
                # Visualizations
                st.subheader("Predictions vs Actual Values")
                
                # Training data visualization
                train_fig = Visualizer.plot_predictions(
                    train_df.index[params['seq_length']:],
                    y_train[params['seq_length']:],
                    train_pred,
                    "Training Data: Actual vs Predicted"
                )
                st.plotly_chart(train_fig, use_container_width=True)
                
                # Test data visualization
                test_fig = Visualizer.plot_predictions(
                    test_df.index[params['seq_length']:],
                    y_test[params['seq_length']:],
                    test_pred,
                    "Test Data: Actual vs Predicted"
                )
                st.plotly_chart(test_fig, use_container_width=True)
                
                # Weekly resampling and visualization
                train_weekly = DataProcessor.resample_weekly(
                    pd.DataFrame({'actual': y_train[params['seq_length']:], 
                                'predicted': train_pred}, 
                               index=train_df.index[params['seq_length']:])
                )
                test_weekly = DataProcessor.resample_weekly(
                    pd.DataFrame({'actual': y_test[params['seq_length']:], 
                                'predicted': test_pred}, 
                               index=test_df.index[params['seq_length']:])
                )
                
                # Weekly visualizations
                st.subheader("Weekly Predictions")
                train_weekly_fig = Visualizer.plot_predictions(
                    train_weekly.index,
                    train_weekly['actual'].values,
                    train_weekly['predicted'].values,
                    "Training Data: Weekly Actual vs Predicted"
                )
                st.plotly_chart(train_weekly_fig, use_container_width=True)
                
                test_weekly_fig = Visualizer.plot_predictions(
                    test_weekly.index,
                    test_weekly['actual'].values,
                    test_weekly['predicted'].values,
                    "Test Data: Weekly Actual vs Predicted"
                )
                st.plotly_chart(test_weekly_fig, use_container_width=True)
                
                # Download predictions
                st.subheader("Download Predictions")
                
                # Prepare download data
                test_predictions_df = pd.DataFrame({
                    'Date': test_df.index[params['seq_length']:],
                    'Actual': y_test[params['seq_length']:],
                    'Predicted': test_pred
                })
                
                # Create download button
                buffer = io.BytesIO()
                test_predictions_df.to_excel(buffer, index=False)
                buffer.seek(0)
                st.download_button(
                    label="Download Predictions",
                    data=buffer,
                    file_name="predictions.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.info("Please upload both training and test datasets to proceed.")
