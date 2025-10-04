#!/usr/bin/env python3
"""
Demo script for adversarial defense mechanisms.
This script demonstrates the basic functionality of the framework.
"""

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from src.models import create_model
from src.attacks import create_attack
from src.defenses import create_defense
from src.visualization import AdversarialVisualizer


def main():
    """Run a simple demonstration."""
    print("🛡️ Adversarial Defense Mechanisms Demo")
    print("=" * 50)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load a small sample of MNIST data
    print("\n📁 Loading MNIST data...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    
    # Get sample data
    images, labels = next(iter(dataloader))
    images = images.to(device)
    labels = labels.to(device)
    
    print(f"Loaded {len(images)} sample images")
    
    # Create model
    print("\n🏗️ Creating model...")
    model = create_model("modern_cnn").to(device)
    model.eval()
    
    # Test clean accuracy
    with torch.no_grad():
        clean_outputs = model(images)
        clean_preds = clean_outputs.argmax(dim=1)
        clean_acc = (clean_preds == labels).float().mean().item()
    
    print(f"Clean accuracy: {clean_acc:.3f}")
    
    # Test FGSM attack
    print("\n🎯 Testing FGSM attack...")
    fgsm_attack = create_attack("fgsm", model, device, epsilon=0.2)
    adv_images = fgsm_attack.attack(images, labels)
    
    with torch.no_grad():
        adv_outputs = model(adv_images)
        adv_preds = adv_outputs.argmax(dim=1)
        adv_acc = (adv_preds == labels).float().mean().item()
    
    print(f"Adversarial accuracy: {adv_acc:.3f}")
    print(f"Accuracy drop: {clean_acc - adv_acc:.3f}")
    
    # Test defenses
    print("\n🛡️ Testing defenses...")
    
    defenses = [
        ("JPEG Compression", "jpeg", {"quality": 75}),
        ("Gaussian Noise", "gaussian_noise", {"std": 0.1}),
        ("Gaussian Blur", "gaussian_blur", {"sigma": 1.0}),
    ]
    
    defense_results = []
    
    for defense_name, defense_type, defense_kwargs in defenses:
        defense = create_defense(defense_type, device, **defense_kwargs)
        defended_images = defense.defend(adv_images)
        
        with torch.no_grad():
            defended_outputs = model(defended_images)
            defended_preds = defended_outputs.argmax(dim=1)
            defended_acc = (defended_preds == labels).float().mean().item()
        
        improvement = defended_acc - adv_acc
        defense_results.append((defense_name, defended_acc, improvement))
        
        print(f"{defense_name}: {defended_acc:.3f} (improvement: {improvement:+.3f})")
    
    # Visualize results
    print("\n📊 Creating visualizations...")
    
    # Create a simple visualization
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    for i in range(5):
        # Original image
        axes[0, i].imshow(images[i].cpu().squeeze(), cmap='gray')
        axes[0, i].set_title(f"Original\nPred: {clean_preds[i].item()}")
        axes[0, i].axis('off')
        
        # Adversarial image
        axes[1, i].imshow(adv_images[i].cpu().squeeze(), cmap='gray')
        axes[1, i].set_title(f"Adversarial\nPred: {adv_preds[i].item()}")
        axes[1, i].axis('off')
    
    plt.suptitle("Adversarial Examples (FGSM, ε=0.2)")
    plt.tight_layout()
    plt.savefig("demo_adversarial_examples.png", dpi=150, bbox_inches='tight')
    print("Visualization saved as 'demo_adversarial_examples.png'")
    
    # Summary
    print("\n📋 Summary:")
    print(f"Clean accuracy: {clean_acc:.3f}")
    print(f"Adversarial accuracy: {adv_acc:.3f}")
    print("Defense results:")
    for defense_name, acc, improvement in defense_results:
        print(f"  {defense_name}: {acc:.3f} ({improvement:+.3f})")
    
    print("\n✅ Demo completed!")
    print("\nTo run more comprehensive experiments:")
    print("  python main.py --quick-test")
    print("\nTo launch the interactive web UI:")
    print("  streamlit run web_ui.py")


if __name__ == "__main__":
    main()
