# Differentiable Optimization Layers in Machine Learning Pipelines

## Overview
This repository implements an end-to-end differentiable optimization layer integrated directly within a PyTorch neural network pipeline. It allows traditional mathematical programming decisions to backpropagate gradients fluidly across parameters.

## Mathematical Formulation
The layer models a parameterized decision-making process where input feature embeddings $x$ are mapped to a constrained optimal decision vector $y^* \in \mathcal{Y}$. The system computes:

$$y^* = \arg\min_{y \in \mathcal{Y}} \left[ f(x, \theta)^T y + \frac{1}{2} \|y\|_2^2 \right]$$

Subject to the budget and conservation constraints:
$$\sum_{i=1}^n y_i = B, \quad y_i \geq 0$$

## Technical Architecture
- **Framework:** PyTorch (Parametric Auto-differentiation Engine)
- **Layer Integration:** Regularized projection operators embedded into deep learning computational graphs.
- **Application Scope:** Strategic resource allocation, operational research, and constrained decision modeling.
