"""
Test the core homeostatic system functionality.
"""

import pytest
import numpy as np
from src.homunc.core import HomeostaticSystem
from src.homunc.uncertainty import UncertaintyModel


class TestHomeostaticSystem:
    """Test cases for HomeostaticSystem class."""
    
    def test_initialization(self):
        """Test system initialization."""
        system = HomeostaticSystem(target_value=5.0, gain=2.0)
        assert system.target_value == 5.0
        assert system.gain == 2.0
        assert system.current_state == 5.0
        assert system.uncertainty_model is None
        assert len(system.history) == 0
    
    def test_step_without_uncertainty(self):
        """Test single step without uncertainty."""
        system = HomeostaticSystem(target_value=0.0, gain=1.0)
        system.current_state = 1.0  # Start away from target
        
        new_state = system.step()
        
        # Should move toward target
        assert new_state < 1.0
        assert len(system.history) == 1
        assert system.history[0]['state'] == new_state
    
    def test_step_with_disturbance(self):
        """Test step with external disturbance."""
        system = HomeostaticSystem(target_value=0.0, gain=1.0)
        
        state_with_disturbance = system.step(disturbance=0.5)
        system.reset()
        state_without_disturbance = system.step(disturbance=0.0)
        
        # Disturbance should affect the outcome
        assert state_with_disturbance != state_without_disturbance
    
    def test_add_uncertainty(self):
        """Test adding uncertainty to system."""
        system = HomeostaticSystem()
        system.add_uncertainty(noise_level=0.1, uncertainty_type="gaussian")
        
        assert system.uncertainty_model is not None
        assert system.uncertainty_model.magnitude == 0.1
        assert system.uncertainty_model.uncertainty_type == "gaussian"
    
    def test_simulation(self):
        """Test running a simulation."""
        system = HomeostaticSystem(target_value=0.0, gain=1.0)
        system.current_state = 1.0
        
        results = system.simulate(steps=10)
        
        assert 'states' in results
        assert 'history' in results
        assert 'final_state' in results
        assert 'target' in results
        assert len(results['states']) == 10
        assert len(results['history']) == 10
        assert results['target'] == 0.0
    
    def test_reset(self):
        """Test system reset functionality."""
        system = HomeostaticSystem(target_value=5.0)
        system.current_state = 10.0
        system.step()  # Add to history
        
        system.reset()
        
        assert system.current_state == 5.0
        assert len(system.history) == 0
    
    def test_analyze(self):
        """Test system analysis."""
        system = HomeostaticSystem(target_value=0.0, gain=1.0)
        system.current_state = 1.0
        
        # Run some steps
        for _ in range(5):
            system.step()
        
        results = system.analyze()
        
        assert 'stability' in results
        assert 'error_variance' in results
        assert 'mean_absolute_error' in results
        assert 'final_error' in results
        assert isinstance(results['stability'], float)
        assert results['stability'] >= 0.0
    
    def test_convergence(self):
        """Test that system converges to target."""
        system = HomeostaticSystem(target_value=0.0, gain=1.0)
        system.current_state = 2.0  # Start away from target
        
        # Run simulation
        results = system.simulate(steps=50)
        
        # Check convergence
        final_error = abs(results['final_state'] - system.target_value)
        assert final_error < 0.1  # Should be close to target