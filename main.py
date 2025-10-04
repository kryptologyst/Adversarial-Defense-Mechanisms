"""
Main application for adversarial defense experiments.
"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from datetime import datetime

from src.models import create_model
from src.training import ExperimentRunner
from src.database import ExperimentDatabase
from src.visualization import AdversarialVisualizer


class AdversarialDefenseApp:
    """Main application class for adversarial defense experiments."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = self._load_config(config_path)
        self.db = ExperimentDatabase(self.config.get('database_path', 'experiments.db'))
        self.visualizer = AdversarialVisualizer(self.device)
        
        print(f"🚀 Adversarial Defense App initialized on {self.device}")
        print(f"📊 Database: {self.config.get('database_path', 'experiments.db')}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            'model_type': 'modern_cnn',
            'epochs': 10,
            'batch_size': 64,
            'learning_rate': 0.001,
            'attack_types': ['fgsm', 'pgd', 'bim'],
            'defense_types': ['jpeg', 'gaussian_noise', 'gaussian_blur'],
            'epsilon_values': [0.1, 0.2, 0.3],
            'use_adversarial_training': True,
            'adversarial_training_epsilon': 0.2,
            'adversarial_training_attack': 'fgsm',
            'data_dir': './data',
            'results_dir': './results',
            'database_path': 'experiments.db',
            'save_models': True,
            'visualize_results': True
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    user_config = yaml.safe_load(f)
                else:
                    user_config = json.load(f)
            
            # Merge with defaults
            default_config.update(user_config)
        
        return default_config
    
    def prepare_data(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare MNIST dataset."""
        print("📁 Preparing MNIST dataset...")
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Create directories
        data_dir = Path(self.config['data_dir'])
        data_dir.mkdir(exist_ok=True)
        
        # Load datasets
        train_dataset = datasets.MNIST(
            str(data_dir), train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            str(data_dir), train=False, transform=transform
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True,
            num_workers=2
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False,
            num_workers=2
        )
        
        # Create validation split
        val_size = int(0.1 * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=2
        )
        
        print(f"✅ Dataset prepared: {len(train_subset)} train, {len(val_subset)} val, {len(test_dataset)} test")
        
        return train_loader, val_loader, test_loader
    
    def run_experiment(self, experiment_name: str, description: str = "") -> int:
        """
        Run a complete adversarial defense experiment.
        
        Args:
            experiment_name: Name of the experiment
            description: Description of the experiment
        
        Returns:
            Experiment ID
        """
        print(f"\n🔬 Starting experiment: {experiment_name}")
        print(f"📝 Description: {description}")
        
        # Create experiment record
        experiment_id = self.db.create_experiment(
            name=experiment_name,
            description=description,
            model_type=self.config['model_type'],
            attack_types=self.config['attack_types'],
            defense_types=self.config['defense_types'],
            epsilon_values=self.config['epsilon_values'],
            config=self.config
        )
        
        # Prepare data
        train_loader, val_loader, test_loader = self.prepare_data()
        
        # Create model
        print(f"🏗️ Creating {self.config['model_type']} model...")
        model = create_model(self.config['model_type'])
        
        # Initialize experiment runner
        runner = ExperimentRunner(self.device, self.config)
        
        # Train model
        print("🎯 Training model...")
        metrics = runner.train_model(
            model, train_loader, val_loader,
            use_adversarial_training=self.config['use_adversarial_training']
        )
        
        # Save training metrics
        for epoch, (train_loss, train_acc) in enumerate(zip(metrics['train_loss'], metrics['train_acc'])):
            val_loss = metrics['val_loss'][epoch] if 'val_loss' in metrics else None
            val_acc = metrics['val_acc'][epoch] if 'val_acc' in metrics else None
            
            self.db.add_model_performance(
                experiment_id, epoch + 1, train_loss, train_acc, val_loss, val_acc
            )
        
        # Save model if requested
        if self.config['save_models']:
            results_dir = Path(self.config['results_dir'])
            results_dir.mkdir(exist_ok=True)
            model_path = results_dir / f"{experiment_name}_model.pth"
            torch.save(model.state_dict(), model_path)
            print(f"💾 Model saved to {model_path}")
        
        # Evaluate robustness
        print("🛡️ Evaluating robustness...")
        robustness_results = runner.evaluate_robustness(
            model, test_loader,
            attack_types=self.config['attack_types'],
            defense_types=self.config['defense_types'],
            epsilon_values=self.config['epsilon_values']
        )
        
        # Save results to database
        for attack_type, attack_results in robustness_results.items():
            if attack_type == 'clean':
                continue
            
            for key, accuracy in attack_results.items():
                if 'no_defense' in key:
                    epsilon = float(key.split('_')[1])
                    self.db.add_result(
                        experiment_id, attack_type, epsilon, None, accuracy
                    )
                else:
                    parts = key.split('_')
                    epsilon = float(parts[1])
                    defense_type = parts[2]
                    self.db.add_result(
                        experiment_id, attack_type, epsilon, defense_type, accuracy
                    )
        
        # Save results to file
        results_dir = Path(self.config['results_dir'])
        results_dir.mkdir(exist_ok=True)
        results_file = results_dir / f"{experiment_name}_results.json"
        runner.save_results(robustness_results, str(results_file))
        
        # Export database data
        db_export_file = results_dir / f"{experiment_name}_database.json"
        self.db.export_experiment_data(experiment_id, str(db_export_file))
        
        # Visualize results if requested
        if self.config['visualize_results']:
            print("📊 Creating visualizations...")
            
            # Get some test samples for visualization
            test_samples = next(iter(test_loader))
            sample_images = test_samples[0][:5]
            sample_labels = test_samples[1][:5]
            
            # Visualize adversarial examples
            viz_file = results_dir / f"{experiment_name}_adversarial_examples.png"
            self.visualizer.visualize_adversarial_examples(
                model, sample_images, sample_labels,
                attack_types=self.config['attack_types'][:2],  # Limit for visualization
                epsilon_values=self.config['epsilon_values'],
                save_path=str(viz_file)
            )
            
            # Visualize defense effectiveness
            defense_viz_file = results_dir / f"{experiment_name}_defense_effectiveness.png"
            self.visualizer.visualize_defense_effectiveness(
                robustness_results, save_path=str(defense_viz_file)
            )
            
            # Plot training curves
            training_viz_file = results_dir / f"{experiment_name}_training_curves.png"
            self.visualizer.plot_training_curves(metrics, save_path=str(training_viz_file))
            
            # Create summary report
            report_file = results_dir / f"{experiment_name}_summary.txt"
            summary = self.visualizer.create_summary_report(
                robustness_results, save_path=str(report_file)
            )
            print("\n" + summary)
        
        print(f"✅ Experiment {experiment_name} completed!")
        print(f"📊 Results saved to {results_dir}")
        
        return experiment_id
    
    def run_quick_test(self) -> int:
        """Run a quick test experiment with minimal configuration."""
        print("⚡ Running quick test experiment...")
        
        # Override config for quick test
        quick_config = self.config.copy()
        quick_config.update({
            'epochs': 2,
            'attack_types': ['fgsm'],
            'defense_types': ['jpeg'],
            'epsilon_values': [0.2],
            'batch_size': 32
        })
        
        # Temporarily update config
        original_config = self.config
        self.config = quick_config
        
        try:
            experiment_id = self.run_experiment(
                "quick_test", 
                "Quick test experiment with minimal configuration"
            )
            return experiment_id
        finally:
            # Restore original config
            self.config = original_config
    
    def list_experiments(self):
        """List all experiments in the database."""
        experiments = self.db.get_all_experiments()
        
        if not experiments:
            print("📭 No experiments found in database.")
            return
        
        print(f"\n📋 Found {len(experiments)} experiments:")
        print("-" * 80)
        
        for exp in experiments:
            print(f"ID: {exp['id']}")
            print(f"Name: {exp['name']}")
            print(f"Description: {exp['description']}")
            print(f"Model: {exp['model_type']}")
            print(f"Created: {exp['created_at']}")
            print("-" * 80)
    
    def show_experiment_details(self, experiment_id: int):
        """Show detailed results for an experiment."""
        experiment = self.db.get_experiment(experiment_id)
        if not experiment:
            print(f"❌ Experiment {experiment_id} not found.")
            return
        
        print(f"\n🔍 Experiment Details: {experiment['name']}")
        print("=" * 60)
        print(f"Description: {experiment['description']}")
        print(f"Model Type: {experiment['model_type']}")
        print(f"Attack Types: {', '.join(experiment['attack_types'])}")
        print(f"Defense Types: {', '.join(experiment['defense_types'])}")
        print(f"Epsilon Values: {experiment['epsilon_values']}")
        print(f"Created: {experiment['created_at']}")
        
        # Get results
        results = self.db.get_experiment_results(experiment_id)
        if results:
            print(f"\n📊 Results Summary:")
            print("-" * 40)
            
            # Group by attack type
            attack_results = {}
            for result in results:
                attack_type = result['attack_type']
                if attack_type not in attack_results:
                    attack_results[attack_type] = []
                attack_results[attack_type].append(result)
            
            for attack_type, attack_results_list in attack_results.items():
                print(f"\n{attack_type.upper()} Attack:")
                for result in attack_results_list:
                    defense = result['defense_type'] or 'No Defense'
                    print(f"  ε={result['epsilon']:.1f}, {defense}: {result['accuracy']:.4f}")
        
        # Get best defenses
        best_defenses = self.db.get_best_defenses(experiment_id)
        if best_defenses:
            print(f"\n🏆 Best Defenses:")
            print("-" * 40)
            for key, defense in best_defenses.items():
                print(f"{defense['attack_type']} (ε={defense['epsilon']:.1f}): "
                      f"{defense['defense_type']} ({defense['accuracy']:.4f})")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Adversarial Defense Experiments')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--experiment', type=str, help='Experiment name')
    parser.add_argument('--description', type=str, default='', help='Experiment description')
    parser.add_argument('--quick-test', action='store_true', help='Run quick test experiment')
    parser.add_argument('--list', action='store_true', help='List all experiments')
    parser.add_argument('--show', type=int, help='Show details for experiment ID')
    
    args = parser.parse_args()
    
    # Initialize app
    app = AdversarialDefenseApp(args.config)
    
    if args.list:
        app.list_experiments()
    elif args.show:
        app.show_experiment_details(args.show)
    elif args.quick_test:
        app.run_quick_test()
    else:
        experiment_name = args.experiment or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        app.run_experiment(experiment_name, args.description)


if __name__ == "__main__":
    main()
