"""
Visualization utilities for adversarial defense experiments.
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import seaborn as sns
from pathlib import Path
import json

from .attacks import create_attack
from .defenses import create_defense


class AdversarialVisualizer:
    """Class for visualizing adversarial examples and defense effectiveness."""
    
    def __init__(self, device: torch.device):
        self.device = device
    
    def visualize_adversarial_examples(self, model: torch.nn.Module, 
                                     images: torch.Tensor, labels: torch.Tensor,
                                     attack_types: List[str] = None,
                                     epsilon_values: List[float] = None,
                                     save_path: Optional[str] = None) -> None:
        """
        Visualize adversarial examples for different attacks and epsilon values.
        
        Args:
            model: Target model
            images: Input images
            labels: True labels
            attack_types: List of attack types
            epsilon_values: List of epsilon values
            save_path: Path to save the plot
        """
        if attack_types is None:
            attack_types = ['fgsm', 'pgd']
        
        if epsilon_values is None:
            epsilon_values = [0.1, 0.2, 0.3]
        
        num_images = min(5, images.size(0))
        num_attacks = len(attack_types)
        num_epsilons = len(epsilon_values)
        
        fig, axes = plt.subplots(num_images, num_attacks * num_epsilons + 1, 
                                figsize=(4 * (num_attacks * num_epsilons + 1), 3 * num_images))
        
        if num_images == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_images):
            # Original image
            img = images[i].cpu().squeeze().numpy()
            axes[i, 0].imshow(img, cmap='gray')
            axes[i, 0].set_title(f'Original\nLabel: {labels[i].item()}')
            axes[i, 0].axis('off')
            
            col_idx = 1
            
            for attack_type in attack_types:
                for epsilon in epsilon_values:
                    # Generate adversarial example
                    attack = create_attack(attack_type, model, self.device, epsilon=epsilon)
                    adv_img = attack.attack(images[i:i+1], labels[i:i+1])
                    
                    # Get prediction
                    with torch.no_grad():
                        pred = model(adv_img).argmax(dim=1).item()
                    
                    # Display adversarial image
                    adv_img_np = adv_img[0].cpu().squeeze().numpy()
                    axes[i, col_idx].imshow(adv_img_np, cmap='gray')
                    axes[i, col_idx].set_title(f'{attack_type.upper()}\nε={epsilon}\nPred: {pred}')
                    axes[i, col_idx].axis('off')
                    
                    col_idx += 1
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Adversarial examples visualization saved to {save_path}")
        
        plt.show()
    
    def visualize_defense_effectiveness(self, results: Dict[str, Any], 
                                      save_path: Optional[str] = None) -> None:
        """
        Visualize defense effectiveness against different attacks.
        
        Args:
            results: Results dictionary from robustness evaluation
            save_path: Path to save the plot
        """
        # Extract data for visualization
        attack_types = []
        epsilon_values = []
        defense_types = []
        accuracies = []
        
        for attack_type, attack_results in results.items():
            if attack_type == 'clean':
                continue
            
            for key, accuracy in attack_results.items():
                if 'no_defense' in key:
                    parts = key.split('_')
                    epsilon = float(parts[1])
                    
                    attack_types.append(attack_type)
                    epsilon_values.append(epsilon)
                    defense_types.append('No Defense')
                    accuracies.append(accuracy)
                
                elif any(defense in key for defense in ['jpeg', 'gaussian_noise', 'gaussian_blur']):
                    parts = key.split('_')
                    epsilon = float(parts[1])
                    defense = parts[2]
                    
                    attack_types.append(attack_type)
                    epsilon_values.append(epsilon)
                    defense_types.append(defense.replace('_', ' ').title())
                    accuracies.append(accuracy)
        
        # Create DataFrame for easier plotting
        import pandas as pd
        df = pd.DataFrame({
            'Attack': attack_types,
            'Epsilon': epsilon_values,
            'Defense': defense_types,
            'Accuracy': accuracies
        })
        
        # Create subplots for each attack type
        attack_types_unique = df['Attack'].unique()
        fig, axes = plt.subplots(1, len(attack_types_unique), 
                                figsize=(5 * len(attack_types_unique), 6))
        
        if len(attack_types_unique) == 1:
            axes = [axes]
        
        for i, attack_type in enumerate(attack_types_unique):
            attack_data = df[df['Attack'] == attack_type]
            
            # Pivot table for heatmap
            pivot_data = attack_data.pivot(index='Defense', columns='Epsilon', values='Accuracy')
            
            # Create heatmap
            sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                       vmin=0, vmax=1, ax=axes[i], cbar_kws={'label': 'Accuracy'})
            axes[i].set_title(f'{attack_type.upper()} Attack')
            axes[i].set_xlabel('Epsilon')
            axes[i].set_ylabel('Defense Method')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Defense effectiveness visualization saved to {save_path}")
        
        plt.show()
    
    def plot_training_curves(self, metrics: Dict[str, List[float]], 
                            save_path: Optional[str] = None) -> None:
        """
        Plot training curves.
        
        Args:
            metrics: Dictionary containing training metrics
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        epochs = range(1, len(metrics['train_loss']) + 1)
        
        # Loss curves
        axes[0].plot(epochs, metrics['train_loss'], 'b-', label='Training Loss')
        if 'val_loss' in metrics:
            axes[0].plot(epochs, metrics['val_loss'], 'r-', label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy curves
        axes[1].plot(epochs, metrics['train_acc'], 'b-', label='Training Accuracy')
        if 'val_acc' in metrics:
            axes[1].plot(epochs, metrics['val_acc'], 'r-', label='Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        
        plt.show()
    
    def visualize_perturbation_magnitude(self, original_images: torch.Tensor,
                                       adversarial_images: torch.Tensor,
                                       save_path: Optional[str] = None) -> None:
        """
        Visualize perturbation magnitude between original and adversarial images.
        
        Args:
            original_images: Original images
            adversarial_images: Adversarial images
            save_path: Path to save the plot
        """
        num_images = min(5, original_images.size(0))
        
        fig, axes = plt.subplots(3, num_images, figsize=(3 * num_images, 9))
        
        for i in range(num_images):
            # Original image
            orig_img = original_images[i].cpu().squeeze().numpy()
            axes[0, i].imshow(orig_img, cmap='gray')
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')
            
            # Adversarial image
            adv_img = adversarial_images[i].cpu().squeeze().numpy()
            axes[1, i].imshow(adv_img, cmap='gray')
            axes[1, i].set_title(f'Adversarial {i+1}')
            axes[1, i].axis('off')
            
            # Perturbation
            perturbation = adv_img - orig_img
            im = axes[2, i].imshow(perturbation, cmap='RdBu_r', vmin=-0.3, vmax=0.3)
            axes[2, i].set_title(f'Perturbation {i+1}')
            axes[2, i].axis('off')
        
        # Add colorbar for perturbation
        fig.colorbar(im, ax=axes[2, :], orientation='horizontal', 
                    label='Perturbation Magnitude', pad=0.1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Perturbation visualization saved to {save_path}")
        
        plt.show()
    
    def create_summary_report(self, results: Dict[str, Any], 
                            save_path: Optional[str] = None) -> str:
        """
        Create a text summary report of the results.
        
        Args:
            results: Results dictionary
            save_path: Path to save the report
        
        Returns:
            Summary report as string
        """
        report = []
        report.append("=" * 60)
        report.append("ADVERSARIAL DEFENSE EXPERIMENT SUMMARY")
        report.append("=" * 60)
        
        # Clean accuracy
        if 'clean' in results:
            clean_acc = results['clean']['accuracy']
            report.append(f"\nClean Accuracy: {clean_acc:.4f} ({clean_acc*100:.2f}%)")
        
        # Attack results
        report.append("\n" + "-" * 40)
        report.append("ATTACK RESULTS")
        report.append("-" * 40)
        
        for attack_type, attack_results in results.items():
            if attack_type == 'clean':
                continue
            
            report.append(f"\n{attack_type.upper()} Attack:")
            
            # Group by epsilon
            epsilon_results = {}
            for key, accuracy in attack_results.items():
                if 'epsilon_' in key:
                    epsilon = key.split('_')[1]
                    if epsilon not in epsilon_results:
                        epsilon_results[epsilon] = {}
                    
                    if 'no_defense' in key:
                        epsilon_results[epsilon]['no_defense'] = accuracy
                    else:
                        defense = key.split('_')[2]
                        epsilon_results[epsilon][defense] = accuracy
            
            for epsilon, defenses in epsilon_results.items():
                report.append(f"  ε = {epsilon}:")
                for defense, accuracy in defenses.items():
                    defense_name = defense.replace('_', ' ').title()
                    report.append(f"    {defense_name}: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Best defenses
        report.append("\n" + "-" * 40)
        report.append("BEST DEFENSES BY ATTACK")
        report.append("-" * 40)
        
        for attack_type, attack_results in results.items():
            if attack_type == 'clean':
                continue
            
            best_defenses = {}
            for key, accuracy in attack_results.items():
                if 'epsilon_' in key and 'no_defense' not in key:
                    epsilon = key.split('_')[1]
                    defense = key.split('_')[2]
                    
                    if epsilon not in best_defenses or accuracy > best_defenses[epsilon][1]:
                        best_defenses[epsilon] = (defense, accuracy)
            
            report.append(f"\n{attack_type.upper()} Attack:")
            for epsilon, (defense, accuracy) in best_defenses.items():
                defense_name = defense.replace('_', ' ').title()
                report.append(f"  ε = {epsilon}: {defense_name} ({accuracy:.4f})")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Summary report saved to {save_path}")
        
        return report_text
