"""
Training and evaluation utilities for adversarial defense experiments.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from tqdm import tqdm
import time
from pathlib import Path

from .models import create_model
from .attacks import create_attack
from .defenses import create_defense, AdversarialTraining, DefensiveDistillation


class ExperimentRunner:
    """Main class for running adversarial defense experiments."""
    
    def __init__(self, device: torch.device, config: Dict[str, Any]):
        self.device = device
        self.config = config
        self.results = {}
    
    def train_model(self, model: nn.Module, train_loader: DataLoader, 
                   val_loader: Optional[DataLoader] = None,
                   use_adversarial_training: bool = False) -> Dict[str, List[float]]:
        """
        Train a model with optional adversarial training.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            use_adversarial_training: Whether to use adversarial training
        
        Returns:
            Dictionary containing training metrics
        """
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.get('learning_rate', 0.001))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        
        metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        adversarial_trainer = None
        if use_adversarial_training:
            adversarial_trainer = AdversarialTraining(
                model, self.device, 
                self.config.get('attack_type', 'fgsm'),
                self.config.get('epsilon', 0.2)
            )
        
        epochs = self.config.get('epochs', 10)
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                if adversarial_trainer:
                    loss = adversarial_trainer.train_step(images, labels, optimizer)
                else:
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = F.cross_entropy(outputs, labels)
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss if isinstance(loss, float) else loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({
                    'Loss': f'{train_loss/(train_total//len(train_loader)):.4f}',
                    'Acc': f'{100*train_correct/train_total:.2f}%'
                })
            
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            
            metrics['train_loss'].append(train_loss)
            metrics['train_acc'].append(train_acc)
            
            # Validation phase
            if val_loader:
                val_loss, val_acc = self.evaluate_model(model, val_loader)
                metrics['val_loss'].append(val_loss)
                metrics['val_acc'].append(val_acc)
                
                print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            else:
                print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            
            scheduler.step()
        
        return metrics
    
    def evaluate_model(self, model: nn.Module, test_loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate model on test data.
        
        Args:
            model: Model to evaluate
            test_loader: Test data loader
        
        Returns:
            Tuple of (loss, accuracy)
        """
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(test_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def evaluate_robustness(self, model: nn.Module, test_loader: DataLoader,
                           attack_types: List[str] = None,
                           defense_types: List[str] = None,
                           epsilon_values: List[float] = None) -> Dict[str, Any]:
        """
        Evaluate model robustness against various attacks and defenses.
        
        Args:
            model: Model to evaluate
            test_loader: Test data loader
            attack_types: List of attack types to test
            defense_types: List of defense types to test
            epsilon_values: List of epsilon values to test
        
        Returns:
            Dictionary containing robustness results
        """
        if attack_types is None:
            attack_types = ['fgsm', 'pgd', 'bim']
        
        if defense_types is None:
            defense_types = ['jpeg', 'gaussian_noise', 'gaussian_blur']
        
        if epsilon_values is None:
            epsilon_values = [0.1, 0.2, 0.3]
        
        results = {}
        
        # Clean accuracy
        clean_loss, clean_acc = self.evaluate_model(model, test_loader)
        results['clean'] = {'loss': clean_loss, 'accuracy': clean_acc}
        
        print(f"Clean accuracy: {clean_acc:.4f}")
        
        # Test each attack
        for attack_type in attack_types:
            results[attack_type] = {}
            
            for epsilon in epsilon_values:
                print(f"\nTesting {attack_type.upper()} attack with epsilon={epsilon}")
                
                # Create attack
                attack = create_attack(attack_type, model, self.device, epsilon=epsilon)
                
                # Test without defense
                acc_no_defense = self._test_attack_with_defense(
                    model, test_loader, attack, None
                )
                results[attack_type][f'epsilon_{epsilon}_no_defense'] = acc_no_defense
                
                print(f"  No defense: {acc_no_defense:.4f}")
                
                # Test with each defense
                for defense_type in defense_types:
                    defense = create_defense(defense_type, self.device)
                    acc_with_defense = self._test_attack_with_defense(
                        model, test_loader, attack, defense
                    )
                    results[attack_type][f'epsilon_{epsilon}_{defense_type}'] = acc_with_defense
                    
                    print(f"  {defense_type}: {acc_with_defense:.4f}")
        
        return results
    
    def _test_attack_with_defense(self, model: nn.Module, test_loader: DataLoader,
                                 attack, defense=None, num_samples: int = 1000) -> float:
        """
        Test model against attack with optional defense.
        
        Args:
            model: Model to test
            test_loader: Test data loader
            attack: Attack instance
            defense: Defense instance (optional)
            num_samples: Number of samples to test
        
        Returns:
            Accuracy under attack
        """
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                if total >= num_samples:
                    break
                
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Generate adversarial examples
                adv_images = attack.attack(images, labels)
                
                # Apply defense if specified
                if defense:
                    adv_images = defense.defend(adv_images)
                
                # Evaluate
                outputs = model(adv_images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total if total > 0 else 0.0
    
    def run_defensive_distillation(self, teacher_model: nn.Module, 
                                  train_loader: DataLoader,
                                  student_model_type: str = "modern_cnn") -> nn.Module:
        """
        Run defensive distillation experiment.
        
        Args:
            teacher_model: Pre-trained teacher model
            train_loader: Training data loader
            student_model_type: Type of student model to create
        
        Returns:
            Trained student model
        """
        # Create student model
        student_model = create_model(student_model_type).to(self.device)
        optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
        
        # Run defensive distillation
        distiller = DefensiveDistillation(teacher_model, student_model, self.device)
        losses = distiller.train_student(train_loader, optimizer, epochs=5)
        
        return student_model
    
    def save_results(self, results: Dict[str, Any], filepath: str):
        """Save results to file."""
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        results_serializable = convert_numpy(results)
        
        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """Load results from file."""
        import json
        
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        return results
