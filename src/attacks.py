"""
Advanced adversarial attack implementations.
"""
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Union
from torch import Tensor
import numpy as np


class AdversarialAttack:
    """Base class for adversarial attacks."""
    
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device
    
    def attack(self, images: Tensor, labels: Tensor, **kwargs) -> Tensor:
        """Generate adversarial examples."""
        raise NotImplementedError


class FGSMAttack(AdversarialAttack):
    """Fast Gradient Sign Method attack."""
    
    def __init__(self, model: torch.nn.Module, device: torch.device, epsilon: float = 0.2):
        super().__init__(model, device)
        self.epsilon = epsilon
    
    def attack(self, images: Tensor, labels: Tensor, epsilon: Optional[float] = None) -> Tensor:
        """
        Generate FGSM adversarial examples.
        
        Args:
            images: Input images
            labels: True labels
            epsilon: Attack strength (overrides default)
        
        Returns:
            Adversarial examples
        """
        epsilon = epsilon or self.epsilon
        
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        images.requires_grad = True
        
        # Forward pass
        outputs = self.model(images)
        loss = F.cross_entropy(outputs, labels)
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # Generate adversarial examples
        data_grad = images.grad.data
        adv_images = images + epsilon * data_grad.sign()
        adv_images = torch.clamp(adv_images, 0, 1)
        
        return adv_images.detach()


class PGDAttack(AdversarialAttack):
    """Projected Gradient Descent attack."""
    
    def __init__(self, model: torch.nn.Module, device: torch.device, 
                 epsilon: float = 0.2, alpha: float = 0.01, steps: int = 40):
        super().__init__(model, device)
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
    
    def attack(self, images: Tensor, labels: Tensor, 
               epsilon: Optional[float] = None, 
               alpha: Optional[float] = None,
               steps: Optional[int] = None) -> Tensor:
        """
        Generate PGD adversarial examples.
        
        Args:
            images: Input images
            labels: True labels
            epsilon: Attack strength
            alpha: Step size
            steps: Number of iterations
        
        Returns:
            Adversarial examples
        """
        epsilon = epsilon or self.epsilon
        alpha = alpha or self.alpha
        steps = steps or self.steps
        
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        # Initialize adversarial examples
        adv_images = images.clone().detach()
        
        for _ in range(steps):
            adv_images.requires_grad = True
            
            # Forward pass
            outputs = self.model(adv_images)
            loss = F.cross_entropy(outputs, labels)
            
            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Update adversarial examples
            data_grad = adv_images.grad.data
            adv_images = adv_images + alpha * data_grad.sign()
            
            # Project to epsilon ball
            delta = torch.clamp(adv_images - images, -epsilon, epsilon)
            adv_images = torch.clamp(images + delta, 0, 1).detach()
        
        return adv_images


class CWAttack(AdversarialAttack):
    """Carlini & Wagner attack."""
    
    def __init__(self, model: torch.nn.Module, device: torch.device, 
                 c: float = 1.0, kappa: float = 0.0, steps: int = 1000):
        super().__init__(model, device)
        self.c = c
        self.kappa = kappa
        self.steps = steps
    
    def attack(self, images: Tensor, labels: Tensor, 
               c: Optional[float] = None,
               kappa: Optional[float] = None,
               steps: Optional[int] = None) -> Tensor:
        """
        Generate C&W adversarial examples.
        
        Args:
            images: Input images
            labels: True labels
            c: Confidence parameter
            kappa: Margin parameter
            steps: Number of iterations
        
        Returns:
            Adversarial examples
        """
        c = c or self.c
        kappa = kappa or self.kappa
        steps = steps or self.steps
        
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        # Initialize adversarial examples
        adv_images = images.clone().detach()
        adv_images.requires_grad = True
        
        optimizer = torch.optim.Adam([adv_images], lr=0.01)
        
        for step in range(steps):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(adv_images)
            
            # C&W loss
            target_scores = outputs.gather(1, labels.unsqueeze(1))
            max_other_scores = outputs.scatter(1, labels.unsqueeze(1), -float('inf')).max(1)[0]
            
            f = torch.clamp(max_other_scores - target_scores.squeeze() + kappa, min=0)
            loss = f.sum() + c * torch.norm(adv_images - images, p=2)
            
            loss.backward()
            optimizer.step()
            
            # Clamp to valid range
            adv_images.data = torch.clamp(adv_images.data, 0, 1)
        
        return adv_images.detach()


class BIMAttack(AdversarialAttack):
    """Basic Iterative Method attack."""
    
    def __init__(self, model: torch.nn.Module, device: torch.device,
                 epsilon: float = 0.2, alpha: float = 0.01, steps: int = 10):
        super().__init__(model, device)
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
    
    def attack(self, images: Tensor, labels: Tensor,
               epsilon: Optional[float] = None,
               alpha: Optional[float] = None,
               steps: Optional[int] = None) -> Tensor:
        """
        Generate BIM adversarial examples.
        
        Args:
            images: Input images
            labels: True labels
            epsilon: Attack strength
            alpha: Step size
            steps: Number of iterations
        
        Returns:
            Adversarial examples
        """
        epsilon = epsilon or self.epsilon
        alpha = alpha or self.alpha
        steps = steps or self.steps
        
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        # Initialize adversarial examples
        adv_images = images.clone().detach()
        
        for _ in range(steps):
            adv_images.requires_grad = True
            
            # Forward pass
            outputs = self.model(adv_images)
            loss = F.cross_entropy(outputs, labels)
            
            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Update adversarial examples
            data_grad = adv_images.grad.data
            adv_images = adv_images + alpha * data_grad.sign()
            
            # Project to epsilon ball
            delta = torch.clamp(adv_images - images, -epsilon, epsilon)
            adv_images = torch.clamp(images + delta, 0, 1).detach()
        
        return adv_images


def create_attack(attack_type: str, model: torch.nn.Module, device: torch.device, **kwargs) -> AdversarialAttack:
    """
    Factory function to create attack instances.
    
    Args:
        attack_type: Type of attack ("fgsm", "pgd", "cw", "bim")
        model: Target model
        device: Device to run on
        **kwargs: Additional arguments for attack creation
    
    Returns:
        Attack instance
    """
    attacks = {
        "fgsm": FGSMAttack,
        "pgd": PGDAttack,
        "cw": CWAttack,
        "bim": BIMAttack,
    }
    
    if attack_type not in attacks:
        raise ValueError(f"Unknown attack type: {attack_type}. Available: {list(attacks.keys())}")
    
    return attacks[attack_type](model, device, **kwargs)
