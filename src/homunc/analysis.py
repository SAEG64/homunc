"""
Analysis tools for homeostatic systems.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from .core import HomeostaticSystem


class StabilityAnalyzer:
    """
    Analyze stability properties of homeostatic systems.
    """
    
    def __init__(self, system: HomeostaticSystem):
        """
        Initialize analyzer with a homeostatic system.
        
        Args:
            system: The homeostatic system to analyze
        """
        self.system = system
    
    def lyapunov_stability(self, perturbations: List[float], steps: int = 100) -> Dict[str, float]:
        """
        Analyze Lyapunov stability by testing response to perturbations.
        
        Args:
            perturbations: List of initial perturbations to test
            steps: Number of simulation steps
            
        Returns:
            Dictionary with stability metrics
        """
        convergence_times = []
        max_deviations = []
        
        original_state = self.system.current_state
        
        for perturbation in perturbations:
            # Reset system and apply perturbation
            self.system.reset()
            self.system.current_state = self.system.target_value + perturbation
            
            # Simulate
            results = self.system.simulate(steps)
            states = results['states']
            
            # Calculate convergence time (time to get within 1% of target)
            target = self.system.target_value
            tolerance = abs(target * 0.01) if target != 0 else 0.01
            
            convergence_time = steps  # Default if never converges
            for i, state in enumerate(states):
                if abs(state - target) <= tolerance:
                    convergence_time = i
                    break
            
            convergence_times.append(convergence_time)
            max_deviations.append(max(abs(s - target) for s in states))
        
        # Restore original state
        self.system.current_state = original_state
        
        return {
            'mean_convergence_time': np.mean(convergence_times),
            'max_convergence_time': max(convergence_times),
            'mean_max_deviation': np.mean(max_deviations),
            'stability_margin': 1.0 / (1.0 + np.mean(max_deviations))
        }
    
    def frequency_response(self, frequencies: List[float], amplitude: float = 1.0, 
                          steps_per_cycle: int = 20) -> Dict[str, List[float]]:
        """
        Analyze frequency response of the system.
        
        Args:
            frequencies: List of frequencies to test
            amplitude: Amplitude of sinusoidal disturbance
            steps_per_cycle: Number of simulation steps per cycle
            
        Returns:
            Dictionary with frequency response data
        """
        gains = []
        phases = []
        
        original_state = self.system.current_state
        
        for freq in frequencies:
            if freq == 0:
                gains.append(float('inf'))
                phases.append(0.0)
                continue
            
            # Reset system
            self.system.reset()
            
            # Generate sinusoidal disturbance
            cycles = 5  # Simulate 5 cycles
            total_steps = int(cycles * steps_per_cycle)
            dt = 1.0 / (freq * steps_per_cycle)
            
            disturbances = []
            for i in range(total_steps):
                t = i * dt
                disturbances.append(amplitude * np.sin(2 * np.pi * freq * t))
            
            # Simulate
            results = self.system.simulate(total_steps, disturbances)
            states = results['states']
            
            # Analyze last cycle for steady-state response
            last_cycle_start = int(4 * steps_per_cycle)
            last_cycle_states = states[last_cycle_start:]
            last_cycle_disturbances = disturbances[last_cycle_start:]
            
            if len(last_cycle_states) > 0:
                # Calculate gain (ratio of output to input amplitude)
                output_amplitude = (max(last_cycle_states) - min(last_cycle_states)) / 2
                input_amplitude = amplitude
                gain = output_amplitude / input_amplitude if input_amplitude > 0 else 0
                
                # Calculate phase (simplified - would need FFT for accurate phase)
                phase = 0.0  # Placeholder
                
                gains.append(gain)
                phases.append(phase)
            else:
                gains.append(0.0)
                phases.append(0.0)
        
        # Restore original state
        self.system.current_state = original_state
        
        return {
            'frequencies': frequencies,
            'gains': gains,
            'phases': phases,
            'gain_margin': max(gains) / min(g for g in gains if g > 0) if any(g > 0 for g in gains) else 1.0
        }
    
    def uncertainty_impact(self, noise_levels: List[float], trials: int = 10, 
                          steps: int = 100) -> Dict[str, List[float]]:
        """
        Analyze impact of uncertainty on system performance.
        
        Args:
            noise_levels: List of noise levels to test
            trials: Number of trials per noise level
            steps: Number of simulation steps per trial
            
        Returns:
            Dictionary with uncertainty analysis results
        """
        mean_errors = []
        error_variances = []
        stability_metrics = []
        
        original_uncertainty = self.system.uncertainty_model
        original_state = self.system.current_state
        
        for noise_level in noise_levels:
            trial_errors = []
            trial_stabilities = []
            
            for _ in range(trials):
                # Reset and set uncertainty
                self.system.reset()
                self.system.add_uncertainty(noise_level)
                
                # Simulate
                results = self.system.simulate(steps)
                analysis = self.system.analyze()
                
                trial_errors.append(analysis['mean_absolute_error'])
                trial_stabilities.append(analysis['stability'])
            
            mean_errors.append(np.mean(trial_errors))
            error_variances.append(np.var(trial_errors))
            stability_metrics.append(np.mean(trial_stabilities))
        
        # Restore original settings
        self.system.uncertainty_model = original_uncertainty
        self.system.current_state = original_state
        
        return {
            'noise_levels': noise_levels,
            'mean_errors': mean_errors,
            'error_variances': error_variances,
            'stability_metrics': stability_metrics,
            'uncertainty_tolerance': max(n for n, s in zip(noise_levels, stability_metrics) 
                                       if s > 0.5) if any(s > 0.5 for s in stability_metrics) else 0.0
        }