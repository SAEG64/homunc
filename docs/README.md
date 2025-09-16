# Documentation

This directory contains the documentation for the homunc library.

## Overview

Homunc is a Python library for modeling and analyzing uncertainty in homeostatic systems. It provides tools for:

- Creating homeostatic systems with feedback control
- Modeling various types of uncertainty
- Analyzing system stability and performance
- Visualizing system behavior

## Getting Started

See the `examples/` directory for usage examples.

## API Reference

### Core Classes

#### HomeostaticSystem
The main class for creating and simulating homeostatic systems.

#### UncertaintyModel
Models different types of uncertainty that can affect system behavior.

#### StabilityAnalyzer
Provides analysis tools for evaluating system stability and performance.

## Mathematical Background

Homeostatic systems maintain stability through feedback control mechanisms. The basic control law is:

```
u(t) = K * e(t)
```

where:
- u(t) is the control input
- K is the feedback gain
- e(t) is the error (target - current_state)

Uncertainty is modeled as additive noise that can follow different probability distributions (Gaussian, uniform, exponential).

## Installation

```bash
pip install homunc
```

## License

MIT License - see LICENSE file for details.