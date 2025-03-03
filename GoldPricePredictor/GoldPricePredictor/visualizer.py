import plotly.graph_objects as go
import pandas as pd
import numpy as np

class Visualizer:
    @staticmethod
    def plot_training_loss(losses: list) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=losses,
            mode='lines',
            name='Training Loss'
        ))
        fig.update_layout(
            title='Training Loss Over Time',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            template='plotly_white'
        )
        return fig
    
    @staticmethod
    def plot_predictions(dates: pd.DatetimeIndex, y_true: np.ndarray, y_pred: np.ndarray, title: str) -> go.Figure:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=y_true,
            mode='lines',
            name='Actual',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=y_pred,
            mode='lines',
            name='Predicted',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Gold Price',
            template='plotly_white',
            hovermode='x unified'
        )
        
        return fig
