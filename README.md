# Adversarial Defense Mechanisms

A comprehensive Python framework for implementing and evaluating adversarial defense mechanisms against machine learning attacks. This project provides modern implementations of various adversarial attacks and defense strategies with interactive visualization and experiment tracking.

## Features

### Adversarial Attacks
- **FGSM (Fast Gradient Sign Method)** - Fast single-step attack
- **PGD (Projected Gradient Descent)** - Multi-step iterative attack
- **BIM (Basic Iterative Method)** - Iterative variant of FGSM
- **C&W (Carlini & Wagner)** - Optimization-based attack

### Defense Mechanisms
- **JPEG Compression** - Input preprocessing defense
- **Gaussian Noise** - Noise-based defense
- **Gaussian Blur** - Smoothing-based defense
- **Median Filter** - Non-linear filtering defense
- **Bit Depth Reduction** - Quantization defense
- **Adversarial Training** - Training with adversarial examples
- **Defensive Distillation** - Knowledge distillation defense

### Model Architectures
- **Modern CNN** - Enhanced CNN with batch normalization and dropout
- **ResNet** - Residual network architecture for MNIST

### Visualization & Analysis
- Interactive adversarial example visualization
- Defense effectiveness heatmaps
- Training curve plots
- Perturbation magnitude analysis
- Comprehensive experiment reports

### Web Interface
- Streamlit-based interactive UI
- Real-time attack and defense testing
- Experiment comparison tools
- Database explorer

### Experiment Tracking
- SQLite database for results storage
- Comprehensive experiment metadata
- Export/import functionality
- Statistical analysis tools

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Adversarial-Defense-Mechanisms.git
cd Adversarial-Defense-Mechanisms
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic Usage

#### Command Line Interface

Run a quick test experiment:
```bash
python main.py --quick-test
```

Run a full experiment:
```bash
python main.py --experiment "my_experiment" --description "Testing FGSM attacks"
```

Run with custom configuration:
```bash
python main.py --config config.yaml --experiment "custom_experiment"
```

#### Web Interface

Launch the interactive web UI:
```bash
streamlit run web_ui.py
```

Then open your browser to `http://localhost:8501`

#### Python API

```python
from src.models import create_model
from src.attacks import create_attack
from src.defenses import create_defense
from src.training import ExperimentRunner

# Create model
model = create_model("modern_cnn")

# Create attack
attack = create_attack("fgsm", model, device, epsilon=0.2)

# Create defense
defense = create_defense("jpeg", device, quality=75)

# Generate adversarial examples
adv_images = attack.attack(images, labels)

# Apply defense
defended_images = defense.defend(adv_images)
```

## 📁 Project Structure

```
adversarial-defense-mechanisms/
├── src/                          # Source code
│   ├── models.py                 # Model architectures
│   ├── attacks.py                # Adversarial attacks
│   ├── defenses.py               # Defense mechanisms
│   ├── training.py               # Training utilities
│   ├── visualization.py          # Visualization tools
│   └── database.py               # Database utilities
├── tests/                        # Test suite
│   └── test_adversarial_defense.py
├── data/                         # Data directory (auto-created)
├── results/                      # Results directory (auto-created)
├── main.py                       # Main CLI application
├── web_ui.py                     # Streamlit web interface
├── config.yaml                   # Configuration file
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Configuration

The project uses YAML configuration files. Key parameters:

```yaml
# Model Configuration
model_type: "modern_cnn"
epochs: 10
batch_size: 64
learning_rate: 0.001

# Attack Configuration
attack_types: ["fgsm", "pgd", "bim", "cw"]
epsilon_values: [0.1, 0.2, 0.3, 0.4, 0.5]

# Defense Configuration
defense_types: ["jpeg", "gaussian_noise", "gaussian_blur"]
use_adversarial_training: true
```

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## Example Results

### Attack Comparison
| Attack | Epsilon | Clean Acc | Adv Acc | Drop |
|--------|---------|-----------|---------|------|
| FGSM   | 0.1     | 0.95      | 0.85    | 0.10 |
| FGSM   | 0.2     | 0.95      | 0.70    | 0.25 |
| PGD    | 0.1     | 0.95      | 0.80    | 0.15 |
| PGD    | 0.2     | 0.95      | 0.60    | 0.35 |

### Defense Effectiveness
| Defense | FGSM (ε=0.2) | PGD (ε=0.2) | Improvement |
|---------|--------------|-------------|-------------|
| No Defense | 0.70 | 0.60 | - |
| JPEG | 0.75 | 0.65 | +0.05 |
| Gaussian Noise | 0.72 | 0.62 | +0.02 |
| Gaussian Blur | 0.78 | 0.68 | +0.08 |

## Advanced Usage

### Custom Attacks

```python
from src.attacks import AdversarialAttack

class CustomAttack(AdversarialAttack):
    def attack(self, images, labels, **kwargs):
        # Implement your custom attack
        return adversarial_images
```

### Custom Defenses

```python
from src.defenses import DefenseMechanism

class CustomDefense(DefenseMechanism):
    def defend(self, images):
        # Implement your custom defense
        return defended_images
```

### Experiment Tracking

```python
from src.database import ExperimentDatabase

db = ExperimentDatabase()
exp_id = db.create_experiment(
    name="my_experiment",
    description="Custom experiment",
    model_type="modern_cnn",
    attack_types=["fgsm"],
    defense_types=["jpeg"],
    epsilon_values=[0.1, 0.2]
)

# Add results
db.add_result(exp_id, "fgsm", 0.1, None, 0.85)
db.add_result(exp_id, "fgsm", 0.1, "jpeg", 0.90)
```

## Visualization Examples

The framework provides comprehensive visualization capabilities:

- **Adversarial Examples**: Side-by-side comparison of original, adversarial, and defended images
- **Defense Effectiveness**: Heatmaps showing accuracy across different attacks and defenses
- **Training Curves**: Loss and accuracy progression during training
- **Perturbation Analysis**: Visualization of adversarial perturbations

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install black flake8 mypy pytest-cov

# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## References

- Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572.
- Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2017). Towards deep learning models resistant to adversarial attacks. arXiv preprint arXiv:1706.06083.
- Carlini, N., & Wagner, D. (2017). Towards evaluating the robustness of neural networks. 2017 IEEE symposium on security and privacy (SP).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- Streamlit team for the interactive web framework
- The adversarial machine learning research community

## Support

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Contact the maintainers
- Check the documentation


# Adversarial-Defense-Mechanisms
