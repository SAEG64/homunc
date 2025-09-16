"""
Test the uncertainty modeling functionality.
"""

import pytest
import numpy as np
from src.homunc.uncertainty import UncertaintyModel


class TestUncertaintyModel:
    """Test cases for UncertaintyModel class."""
    
    def test_initialization(self):
        """Test uncertainty model initialization."""
        model = UncertaintyModel(magnitude=0.5, uncertainty_type="gaussian")
        assert model.magnitude == 0.5
        assert model.uncertainty_type == "gaussian"
    
    def test_invalid_uncertainty_type(self):
        """Test that invalid uncertainty type raises error."""
        with pytest.raises(ValueError):
            UncertaintyModel(uncertainty_type="invalid")
    
    def test_gaussian_sampling(self):
        """Test gaussian uncertainty sampling."""
        model = UncertaintyModel(magnitude=1.0, uncertainty_type="gaussian")
        
        # Generate many samples
        samples = model.samples(1000)
        
        # Check basic properties
        assert len(samples) == 1000
        assert abs(np.mean(samples)) < 0.2  # Should be close to zero mean
        assert 0.8 < np.std(samples) < 1.2  # Should be close to magnitude
    
    def test_uniform_sampling(self):
        """Test uniform uncertainty sampling."""
        model = UncertaintyModel(magnitude=1.0, uncertainty_type="uniform")
        
        samples = model.samples(1000)
        
        # Check that samples are within bounds
        assert np.all(samples >= -1.0)
        assert np.all(samples <= 1.0)
        assert abs(np.mean(samples)) < 0.1  # Should be close to zero mean
    
    def test_exponential_sampling(self):
        """Test exponential uncertainty sampling."""
        model = UncertaintyModel(magnitude=1.0, uncertainty_type="exponential")
        
        samples = model.samples(1000)
        
        # Check basic properties
        assert len(samples) == 1000
        assert abs(np.mean(samples)) < 0.2  # Should be close to zero mean
    
    def test_single_sample(self):
        """Test single sample generation."""
        model = UncertaintyModel(magnitude=0.5, uncertainty_type="gaussian")
        
        sample = model.sample()
        
        assert isinstance(sample, float)
        # Sample should be reasonable (within 3 standard deviations)
        assert abs(sample) < 3 * 0.5
    
    def test_statistics_gaussian(self):
        """Test theoretical statistics for gaussian distribution."""
        model = UncertaintyModel(magnitude=2.0, uncertainty_type="gaussian")
        
        stats = model.statistics()
        
        assert stats['mean'] == 0.0
        assert stats['variance'] == 4.0  # magnitude^2
        assert stats['std'] == 2.0  # magnitude
    
    def test_statistics_uniform(self):
        """Test theoretical statistics for uniform distribution."""
        model = UncertaintyModel(magnitude=1.0, uncertainty_type="uniform")
        
        stats = model.statistics()
        
        assert stats['mean'] == 0.0
        # Variance for uniform distribution: (2*magnitude)^2 / 12
        expected_variance = (2 * 1.0)**2 / 12
        assert abs(stats['variance'] - expected_variance) < 1e-10
    
    def test_statistics_exponential(self):
        """Test theoretical statistics for exponential distribution."""
        model = UncertaintyModel(magnitude=1.0, uncertainty_type="exponential")
        
        stats = model.statistics()
        
        assert stats['mean'] == 0.0
        assert stats['variance'] == 2.0  # 2 * magnitude^2
        assert abs(stats['std'] - np.sqrt(2)) < 1e-10