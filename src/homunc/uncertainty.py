"""
Uncertainty modeling for homeostatic systems.
"""

import numpy as np
from typing import Union


class UncertaintyModel:
    """
    Model various types of uncertainty in homeostatic systems.
    """
    
    def __init__(self, magnitude: float = 0.1, uncertainty_type: str = "gaussian"):
        """
        Initialize uncertainty model.
        
        Args:
            magnitude: The magnitude/scale of uncertainty
            uncertainty_type: Type of uncertainty distribution
        """
        self.magnitude = magnitude
        self.uncertainty_type = uncertainty_type.lower()
        
        if self.uncertainty_type not in ["gaussian", "uniform", "exponential"]:
            raise ValueError(f"Unsupported uncertainty type: {uncertainty_type}")
    
    def sample(self) -> float:
        """
        Sample from the uncertainty distribution.
        
        Returns:
            A random sample from the uncertainty distribution
        """
        if self.uncertainty_type == "gaussian":
            return np.random.normal(0, self.magnitude)
        elif self.uncertainty_type == "uniform":
            return np.random.uniform(-self.magnitude, self.magnitude)
        elif self.uncertainty_type == "exponential":
            # Symmetric exponential (Laplace-like)
            sign = np.random.choice([-1, 1])
            return sign * np.random.exponential(self.magnitude)
        else:
            return 0.0
    
    def samples(self, n: int) -> np.ndarray:
        """
        Generate multiple samples.
        
        Args:
            n: Number of samples to generate
            
        Returns:
            Array of uncertainty samples
        """
        return np.array([self.sample() for _ in range(n)])
    
    def statistics(self) -> dict:
        """
        Get theoretical statistics of the uncertainty distribution.
        
        Returns:
            Dictionary with mean, variance, and other statistics
        """
        if self.uncertainty_type == "gaussian":
            return {
                "mean": 0.0,
                "variance": self.magnitude**2,
                "std": self.magnitude
            }
        elif self.uncertainty_type == "uniform":
            return {
                "mean": 0.0,
                "variance": (2 * self.magnitude)**2 / 12,
                "std": (2 * self.magnitude) / np.sqrt(12)
            }
        elif self.uncertainty_type == "exponential":
            return {
                "mean": 0.0,
                "variance": 2 * self.magnitude**2,
                "std": np.sqrt(2) * self.magnitude
            }
        else:
            return {"mean": 0.0, "variance": 0.0, "std": 0.0}