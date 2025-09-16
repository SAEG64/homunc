"""
Core homeostatic system implementation.
"""

import numpy as np
from typing import Optional, Dict, Any
from .uncertainty import UncertaintyModel


class HomeostaticSystem:
    """
    A basic homeostatic system with feedback control and uncertainty modeling.
    
    This class represents a system that maintains stability through feedback
    mechanisms while accounting for various sources of uncertainty.
    """
    
    def __init__(self, target_value: float = 0.0, gain: float = 1.0):
        """
        Initialize the homeostatic system.
        
        Args:
            target_value: The desired setpoint for the system
            gain: The feedback gain parameter
        """
        self.target_value = target_value
        self.gain = gain
        self.current_state = target_value
        self.uncertainty_model: Optional[UncertaintyModel] = None
        self.history = []
    
    def add_uncertainty(self, noise_level: float = 0.1, uncertainty_type: str = "gaussian"):
        """
        Add uncertainty to the system.
        
        Args:
            noise_level: The magnitude of uncertainty
            uncertainty_type: Type of uncertainty ("gaussian", "uniform", etc.)
        """
        self.uncertainty_model = UncertaintyModel(noise_level, uncertainty_type)
    
    def step(self, disturbance: float = 0.0) -> float:
        """
        Perform one simulation step.
        
        Args:
            disturbance: External disturbance to the system
            
        Returns:
            The new system state
        """
        # Calculate error
        error = self.target_value - self.current_state
        
        # Apply feedback control
        control_input = self.gain * error
        
        # Apply disturbance
        total_input = control_input + disturbance
        
        # Add uncertainty if present
        if self.uncertainty_model:
            total_input += self.uncertainty_model.sample()
        
        # Update state (simple integration)
        self.current_state += total_input * 0.1  # dt = 0.1
        
        # Store history
        self.history.append({
            'state': self.current_state,
            'error': error,
            'control': control_input,
            'disturbance': disturbance
        })
        
        return self.current_state
    
    def simulate(self, steps: int = 100, disturbances: Optional[list] = None) -> Dict[str, Any]:
        """
        Run a simulation for multiple steps.
        
        Args:
            steps: Number of simulation steps
            disturbances: List of disturbances (optional)
            
        Returns:
            Dictionary containing simulation results
        """
        if disturbances is None:
            disturbances = [0.0] * steps
        
        states = []
        for i in range(steps):
            dist = disturbances[i] if i < len(disturbances) else 0.0
            state = self.step(dist)
            states.append(state)
        
        return {
            'states': states,
            'history': self.history.copy(),
            'final_state': self.current_state,
            'target': self.target_value
        }
    
    def reset(self):
        """Reset the system to initial conditions."""
        self.current_state = self.target_value
        self.history = []
    
    def analyze(self) -> Dict[str, float]:
        """
        Analyze system performance.
        
        Returns:
            Dictionary with analysis results
        """
        if not self.history:
            return {'stability': 0.0, 'error_variance': 0.0}
        
        errors = [h['error'] for h in self.history]
        states = [h['state'] for h in self.history]
        
        # Calculate stability metric (inverse of state variance)
        state_variance = np.var(states) if len(states) > 1 else 0.0
        stability = 1.0 / (1.0 + state_variance)
        
        # Calculate error variance
        error_variance = np.var(errors) if len(errors) > 1 else 0.0
        
        return {
            'stability': stability,
            'error_variance': error_variance,
            'mean_absolute_error': np.mean(np.abs(errors)),
            'final_error': abs(errors[-1]) if errors else 0.0
        }