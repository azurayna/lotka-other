import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import base64
import io

from lotka_volterra import lotka_volterra_reactions, lotka_volterra_propensities
from gillespie import gillespie_simulation
from visualization import plot_simulation_results, plot_multiple_simulations, plot_statistics
from utils import calculate_statistics, generate_csv_download_link

# Page configuration
st.set_page_config(
    page_title="Lotka-Volterra Simulator",
    page_icon="🦊",
    layout="wide"
)

# App title and description
st.title("Lotka-Volterra Predator-Prey Simulation")

with st.expander("About this simulator", expanded=True):
    st.markdown("""
    This application simulates the Lotka-Volterra predator-prey model using the Gillespie algorithm, 
    which is a stochastic algorithm that accounts for the randomness in population dynamics.
    
    ### The Lotka-Volterra Model
    The Lotka-Volterra model describes the dynamics of biological systems in which two species interact, 
    one as a predator and the other as prey. The model consists of a pair of differential equations:
    
    1. **Prey population (X)**: 
        - Increases proportionally to the current population (birth)
        - Decreases due to predation (death)
    
    2. **Predator population (Y)**:
        - Increases proportionally to predation success (birth)
        - Decreases naturally (death)
    
    ### Gillespie Algorithm
    The Gillespie algorithm simulates stochastic trajectories of the system, accounting for the 
    inherent randomness in biological processes. It's particularly useful for systems with small 
    population sizes where deterministic models may not be accurate.
    
    ### Reactions in this model:
    - **Prey birth**: X → X + X (rate: prey_birth_rate)
    - **Predation**: X + Y → Y + Y (rate: predation_rate)
    - **Predator death**: Y → ∅ (rate: predator_death_rate)
    """)

# Create sidebar for parameters
st.sidebar.header("Simulation Parameters")

# Reaction rate parameters
prey_birth_rate = st.sidebar.slider("Prey birth rate (α)", 0.01, 2.0, 1.0, 0.01)
predation_rate = st.sidebar.slider("Predation rate (β)", 0.01, 2.0, 0.1, 0.01)  
predator_death_rate = st.sidebar.slider("Predator death rate (γ)", 0.01, 2.0, 1.0, 0.01)

# Initial populations
st.sidebar.header("Initial Population")
initial_prey = st.sidebar.number_input("Initial prey (X)", min_value=1, max_value=1000, value=100)
initial_predator = st.sidebar.number_input("Initial predator (Y)", min_value=1, max_value=1000, value=50)

# Simulation settings
st.sidebar.header("Simulation Settings")
max_time = st.sidebar.number_input("Maximum simulation time", min_value=1, max_value=1000, value=100)
num_simulations = st.sidebar.number_input("Number of simulations", min_value=1, max_value=50, value=1)

# Run simulation button
if st.sidebar.button("Run Simulation"):
    # Display progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Prepare container for results
    results_container = st.container()
    
    # Initialize lists to store results
    all_results = []
    all_times = []
    all_populations = []
    
    # Run simulations
    for i in range(num_simulations):
        status_text.text(f"Running simulation {i+1}/{num_simulations}")
        
        # Initial state
        initial_state = np.array([initial_prey, initial_predator])
        
        # Run Gillespie simulation
        times, populations = gillespie_simulation(
            initial_state=initial_state,
            propensity_func=lambda x: lotka_volterra_propensities(x, prey_birth_rate, predation_rate, predator_death_rate),
            reaction_func=lotka_volterra_reactions,
            max_time=max_time
        )
        
        all_times.append(times)
        all_populations.append(populations)
        
        # Create a DataFrame for this simulation
        df = pd.DataFrame({
            'Time': times,
            'Prey': populations[:, 0],
            'Predator': populations[:, 1]
        })
        
        all_results.append(df)
        
        # Update progress
        progress_bar.progress((i + 1) / num_simulations)
    
    status_text.text("Simulations completed!")
    
    with results_container:
        if num_simulations == 1:
            st.subheader("Simulation Results")
            
            # Plot single simulation
            fig = plot_simulation_results(all_times[0], all_populations[0])
            st.plotly_chart(fig, use_container_width=True)
            
            # Display statistics
            stats_df = calculate_statistics(all_results[0])
            st.subheader("Statistics")
            st.dataframe(stats_df)
            
            # Download link for simulation data
            st.download_button(
                label="Download Simulation Data (CSV)",
                data=all_results[0].to_csv(index=False).encode('utf-8'),
                file_name=f"lotka_volterra_simulation.csv",
                mime="text/csv"
            )
            
        else:
            st.subheader("Multiple Simulation Results")
            
            # Plot multiple simulations
            multi_fig = plot_multiple_simulations(all_times, all_populations)
            st.plotly_chart(multi_fig, use_container_width=True)
            
            # Calculate and display statistics
            combined_df = pd.concat(all_results, keys=range(len(all_results)))
            combined_df.reset_index(level=0, inplace=True)
            combined_df.rename(columns={'level_0': 'Simulation'}, inplace=True)
            
            # Display statistics for all simulations
            stats_fig = plot_statistics(all_results)
            st.subheader("Statistics Across Simulations")
            st.plotly_chart(stats_fig, use_container_width=True)
            
            # Download link for all simulation data
            st.download_button(
                label="Download All Simulation Data (CSV)",
                data=combined_df.to_csv(index=False).encode('utf-8'),
                file_name=f"lotka_volterra_multiple_simulations.csv",
                mime="text/csv"
            )

# Explanation of parameters
st.sidebar.markdown("""
### Parameter Explanation

**Prey birth rate (α)**: Rate at which prey reproduce  
**Predation rate (β)**: Rate at which predators consume prey and reproduce  
**Predator death rate (γ)**: Natural death rate of predators  

These parameters control the dynamics of the system:
- Higher prey birth rate → More prey
- Higher predation rate → More predator births, more prey deaths
- Higher predator death rate → Fewer predators
""")
