"""
Adversarial Defense Mechanisms Package

A comprehensive framework for implementing and evaluating adversarial defense mechanisms
against machine learning attacks.

Modules:
- models: Neural network architectures
- attacks: Adversarial attack implementations
- defenses: Defense mechanism implementations
- training: Training and evaluation utilities
- visualization: Visualization tools
- database: Database utilities for experiment tracking
"""

__version__ = "1.0.0"
__author__ = "Adversarial Defense Team"
__email__ = "contact@adversarial-defense.com"

# Import main classes for easy access
from .models import create_model, ModernCNN, ResNetMNIST
from .attacks import create_attack, FGSMAttack, PGDAttack, BIMAttack, CWAttack
from .defenses import create_defense, JPEGCompressionDefense, GaussianNoiseDefense
from .training import ExperimentRunner
from .visualization import AdversarialVisualizer
from .database import ExperimentDatabase

__all__ = [
    # Models
    'create_model',
    'ModernCNN',
    'ResNetMNIST',
    
    # Attacks
    'create_attack',
    'FGSMAttack',
    'PGDAttack',
    'BIMAttack',
    'CWAttack',
    
    # Defenses
    'create_defense',
    'JPEGCompressionDefense',
    'GaussianNoiseDefense',
    
    # Training
    'ExperimentRunner',
    
    # Visualization
    'AdversarialVisualizer',
    
    # Database
    'ExperimentDatabase',
]
