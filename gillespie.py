import numpy as np

def gillespie_simulation(initial_state, propensity_func, reaction_func, max_time):
    """
    Implements the Gillespie algorithm for stochastic simulation of chemical or biological systems.
    
    Parameters:
    -----------
    initial_state : numpy.ndarray
        Initial populations of species (e.g., [prey_count, predator_count])
    propensity_func : function
        Function that calculates propensities (rates) of reactions based on current state
    reaction_func : function
        Function that returns the state change for each reaction
    max_time : float
        Maximum time to simulate
    
    Returns:
    --------
    times : numpy.ndarray
        Array of time points
    populations : numpy.ndarray
        Array of population states at each time point
    """
    
    # Initialize
    times = [0]
    current_state = initial_state.copy()
    populations = [current_state.copy()]
    current_time = 0
    
    # Get reaction stoichiometry matrix
    reactions = reaction_func()
    
    # Run simulation
    while current_time < max_time:
        # Calculate propensities
        propensities = propensity_func(current_state)
        
        # Sum of propensities
        propensity_sum = np.sum(propensities)
        
        # If all propensities are zero, system is "dead" - exit loop
        if propensity_sum == 0:
            break
        
        # Sample time to next reaction (exponential distribution)
        time_step = np.random.exponential(scale=1/propensity_sum)
        
        # Ensure we don't exceed max_time
        if current_time + time_step > max_time:
            current_time = max_time
            times.append(current_time)
            populations.append(current_state.copy())
            break
        
        # Update time
        current_time += time_step
        
        # Choose reaction according to propensities
        reaction_probs = propensities / propensity_sum
        reaction_index = np.random.choice(len(propensities), p=reaction_probs)
        
        # Update state according to chosen reaction
        current_state += reactions[reaction_index]
        
        # Ensure no negative populations
        current_state = np.maximum(current_state, 0)
        
        # Store time and state
        times.append(current_time)
        populations.append(current_state.copy())
    
    return np.array(times), np.array(populations)