"""
Basic example demonstrating homeostatic system with uncertainty.

This example shows how to:
1. Create a homeostatic system
2. Add uncertainty
3. Run simulations
4. Analyze results
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from homunc import HomeostaticSystem, StabilityAnalyzer


def basic_homeostasis_example():
    """Demonstrate basic homeostatic system operation."""
    print("=== Basic Homeostasis Example ===")
    
    # Create a homeostatic system
    system = HomeostaticSystem(target_value=10.0, gain=0.5)
    
    # Start away from target
    system.current_state = 15.0
    
    # Run simulation
    results = system.simulate(steps=50)
    
    # Analyze results
    analysis = system.analyze()
    
    print(f"Target value: {system.target_value}")
    print(f"Final state: {results['final_state']:.3f}")
    print(f"Stability: {analysis['stability']:.3f}")
    print(f"Mean absolute error: {analysis['mean_absolute_error']:.3f}")
    
    return results


def uncertainty_example():
    """Demonstrate system with uncertainty."""
    print("\n=== Uncertainty Example ===")
    
    # Create two systems: one with and one without uncertainty
    system_clean = HomeostaticSystem(target_value=0.0, gain=1.0)
    system_noisy = HomeostaticSystem(target_value=0.0, gain=1.0)
    
    # Add uncertainty to second system
    system_noisy.add_uncertainty(noise_level=0.5, uncertainty_type="gaussian")
    
    # Start both systems at same initial condition
    system_clean.current_state = 5.0
    system_noisy.current_state = 5.0
    
    # Run simulations
    results_clean = system_clean.simulate(steps=100)
    results_noisy = system_noisy.simulate(steps=100)
    
    # Analyze both
    analysis_clean = system_clean.analyze()
    analysis_noisy = system_noisy.analyze()
    
    print(f"Clean system stability: {analysis_clean['stability']:.3f}")
    print(f"Noisy system stability: {analysis_noisy['stability']:.3f}")
    print(f"Clean system final error: {analysis_clean['final_error']:.3f}")
    print(f"Noisy system final error: {analysis_noisy['final_error']:.3f}")
    
    return results_clean, results_noisy


def stability_analysis_example():
    """Demonstrate stability analysis."""
    print("\n=== Stability Analysis Example ===")
    
    # Create system
    system = HomeostaticSystem(target_value=0.0, gain=0.8)
    system.add_uncertainty(noise_level=0.1)
    
    # Create analyzer
    analyzer = StabilityAnalyzer(system)
    
    # Test Lyapunov stability
    perturbations = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
    stability_results = analyzer.lyapunov_stability(perturbations, steps=50)
    
    print(f"Mean convergence time: {stability_results['mean_convergence_time']:.1f} steps")
    print(f"Stability margin: {stability_results['stability_margin']:.3f}")
    
    # Test uncertainty impact
    noise_levels = [0.0, 0.1, 0.2, 0.5, 1.0]
    uncertainty_results = analyzer.uncertainty_impact(noise_levels, trials=5, steps=50)
    
    print(f"Uncertainty tolerance: {uncertainty_results['uncertainty_tolerance']:.3f}")
    
    return stability_results, uncertainty_results


def visualization_example():
    """Create visualizations of system behavior."""
    print("\n=== Visualization Example ===")
    
    # Run basic example to get data
    system = HomeostaticSystem(target_value=0.0, gain=1.0)
    system.current_state = 3.0
    
    # Simulate with different uncertainty levels
    results_no_noise = system.simulate(steps=100)
    
    system.reset()
    system.current_state = 3.0
    system.add_uncertainty(noise_level=0.2)
    results_low_noise = system.simulate(steps=100)
    
    system.reset()
    system.current_state = 3.0
    system.add_uncertainty(noise_level=0.5)
    results_high_noise = system.simulate(steps=100)
    
    # Create plots
    plt.figure(figsize=(12, 8))
    
    # Plot 1: System states over time
    plt.subplot(2, 2, 1)
    steps = range(len(results_no_noise['states']))
    plt.plot(steps, results_no_noise['states'], label='No noise', linewidth=2)
    plt.plot(steps, results_low_noise['states'], label='Low noise (0.2)', alpha=0.7)
    plt.plot(steps, results_high_noise['states'], label='High noise (0.5)', alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Target')
    plt.xlabel('Time Steps')
    plt.ylabel('System State')
    plt.title('System Response with Different Noise Levels')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Error over time
    plt.subplot(2, 2, 2)
    errors_no_noise = [h['error'] for h in results_no_noise['history']]
    errors_low_noise = [h['error'] for h in results_low_noise['history']]
    errors_high_noise = [h['error'] for h in results_high_noise['history']]
    
    plt.plot(steps, np.abs(errors_no_noise), label='No noise')
    plt.plot(steps, np.abs(errors_low_noise), label='Low noise')
    plt.plot(steps, np.abs(errors_high_noise), label='High noise')
    plt.xlabel('Time Steps')
    plt.ylabel('Absolute Error')
    plt.title('System Error Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 3: Control effort
    plt.subplot(2, 2, 3)
    control_no_noise = [h['control'] for h in results_no_noise['history']]
    control_low_noise = [h['control'] for h in results_low_noise['history']]
    control_high_noise = [h['control'] for h in results_high_noise['history']]
    
    plt.plot(steps, control_no_noise, label='No noise')
    plt.plot(steps, control_low_noise, label='Low noise')
    plt.plot(steps, control_high_noise, label='High noise')
    plt.xlabel('Time Steps')
    plt.ylabel('Control Input')
    plt.title('Control Effort Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Phase portrait (state vs error)
    plt.subplot(2, 2, 4)
    states_no_noise = results_no_noise['states']
    states_low_noise = results_low_noise['states']
    states_high_noise = results_high_noise['states']
    
    plt.plot(states_no_noise, errors_no_noise, 'o-', label='No noise', markersize=2)
    plt.plot(states_low_noise, errors_low_noise, 'o-', label='Low noise', markersize=2, alpha=0.7)
    plt.plot(states_high_noise, errors_high_noise, 'o-', label='High noise', markersize=2, alpha=0.7)
    plt.xlabel('System State')
    plt.ylabel('Error')
    plt.title('Phase Portrait')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('homeostasis_analysis.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'homeostasis_analysis.png'")
    plt.show()


def main():
    """Run all examples."""
    print("Homunc - Homeostasis Uncertainty Examples")
    print("=" * 40)
    
    # Run examples
    basic_results = basic_homeostasis_example()
    clean_results, noisy_results = uncertainty_example()
    stability_results, uncertainty_results = stability_analysis_example()
    
    # Create visualizations
    try:
        visualization_example()
    except ImportError:
        print("\nNote: matplotlib not available, skipping visualization example")
    except Exception as e:
        print(f"\nVisualization example failed: {e}")
    
    print("\n" + "=" * 40)
    print("All examples completed successfully!")


if __name__ == "__main__":
    main()