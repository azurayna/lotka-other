import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_simulation_results(times, populations):
    """
    Create a plot of prey and predator populations over time.
    
    Parameters:
    -----------
    times : numpy.ndarray
        Array of time points
    populations : numpy.ndarray
        Array of population states at each time point
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Figure with population time series
    """
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("Population Time Series", "Phase Space (Prey vs Predator)"),
        row_heights=[0.7, 0.3],
        vertical_spacing=0.1
    )
    
    # Add population time series
    fig.add_trace(
        go.Scatter(
            x=times, 
            y=populations[:, 0],
            mode='lines',
            name='Prey',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=times, 
            y=populations[:, 1],
            mode='lines',
            name='Predator',
            line=dict(color='red', width=2)
        ),
        row=1, col=1
    )
    
    # Add phase space plot
    fig.add_trace(
        go.Scatter(
            x=populations[:, 0], 
            y=populations[:, 1],
            mode='lines',
            name='Phase Space',
            line=dict(color='green', width=1.5),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Add markers for start and end points in phase space
    fig.add_trace(
        go.Scatter(
            x=[populations[0, 0]],
            y=[populations[0, 1]],
            mode='markers',
            marker=dict(color='green', size=10, symbol='circle-open'),
            name='Start',
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=[populations[-1, 0]],
            y=[populations[-1, 1]],
            mode='markers',
            marker=dict(color='red', size=10, symbol='x'),
            name='End',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=700,
        title='Lotka-Volterra Predator-Prey Simulation',
        legend=dict(orientation="h", y=1.02),
        hovermode='x unified'
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(title_text="Population", row=1, col=1)
    fig.update_xaxes(title_text="Prey Population", row=2, col=1)
    fig.update_yaxes(title_text="Predator Population", row=2, col=1)
    
    return fig

def plot_multiple_simulations(all_times, all_populations):
    """
    Create a plot showing multiple simulation runs.
    
    Parameters:
    -----------
    all_times : list of numpy.ndarray
        List of time arrays for each simulation
    all_populations : list of numpy.ndarray
        List of population arrays for each simulation
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Figure with all simulation runs
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Prey Population", "Predator Population"),
        horizontal_spacing=0.1
    )
    
    num_simulations = len(all_times)
    colors = [f'rgba(31, 119, 180, {0.5 + 0.5 * i / num_simulations})' for i in range(num_simulations)]
    
    for i in range(num_simulations):
        # Prey plot
        fig.add_trace(
            go.Scatter(
                x=all_times[i],
                y=all_populations[i][:, 0],
                mode='lines',
                name=f'Run {i+1} - Prey',
                line=dict(color=colors[i], width=1.5),
                legendgroup=f'sim{i}',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Predator plot
        fig.add_trace(
            go.Scatter(
                x=all_times[i],
                y=all_populations[i][:, 1],
                mode='lines',
                name=f'Run {i+1} - Predator',
                line=dict(color=colors[i], width=1.5, dash='dash'),
                legendgroup=f'sim{i}',
                showlegend=True
            ),
            row=1, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=400,
        title='Multiple Simulation Runs',
        legend=dict(orientation="h", y=1.02),
        hovermode='x unified'
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=1, col=2)
    fig.update_yaxes(title_text="Prey Population", row=1, col=1)
    fig.update_yaxes(title_text="Predator Population", row=1, col=2)
    
    return fig

def plot_statistics(all_results):
    """
    Create box plots showing statistics across multiple simulation runs.
    
    Parameters:
    -----------
    all_results : list of pandas.DataFrame
        List of DataFrames containing simulation results
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Figure with statistical summaries
    """
    # Extract statistics from all simulations
    max_prey = [df['Prey'].max() for df in all_results]
    min_prey = [df['Prey'].min() for df in all_results]
    mean_prey = [df['Prey'].mean() for df in all_results]
    
    max_predator = [df['Predator'].max() for df in all_results]
    min_predator = [df['Predator'].min() for df in all_results]
    mean_predator = [df['Predator'].mean() for df in all_results]
    
    # Create figure
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Prey Statistics", "Predator Statistics"),
        horizontal_spacing=0.1
    )
    
    # Add box plots for prey
    fig.add_trace(
        go.Box(
            y=max_prey,
            name='Maximum',
            boxmean=True,
            marker_color='blue'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Box(
            y=mean_prey,
            name='Mean',
            boxmean=True,
            marker_color='green'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Box(
            y=min_prey,
            name='Minimum',
            boxmean=True,
            marker_color='red'
        ),
        row=1, col=1
    )
    
    # Add box plots for predator
    fig.add_trace(
        go.Box(
            y=max_predator,
            name='Maximum',
            boxmean=True,
            marker_color='blue',
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Box(
            y=mean_predator,
            name='Mean',
            boxmean=True,
            marker_color='green',
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Box(
            y=min_predator,
            name='Minimum',
            boxmean=True,
            marker_color='red',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=400,
        title='Statistical Summary Across Simulations',
        legend=dict(orientation="h", y=1.02)
    )
    
    # Update axes labels
    fig.update_yaxes(title_text="Population", row=1, col=1)
    fig.update_yaxes(title_text="Population", row=1, col=2)
    
    return fig
