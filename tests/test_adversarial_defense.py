"""
Test suite for adversarial defense mechanisms.
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

from src.models import ModernCNN, ResNetMNIST, create_model
from src.attacks import FGSMAttack, PGDAttack, BIMAttack, CWAttack, create_attack
from src.defenses import (
    JPEGCompressionDefense, GaussianNoiseDefense, GaussianBlurDefense,
    MedianFilterDefense, BitDepthReductionDefense, create_defense
)
from src.database import ExperimentDatabase
from src.training import ExperimentRunner


class TestModels:
    """Test model implementations."""
    
    def test_modern_cnn_creation(self):
        """Test ModernCNN model creation."""
        model = ModernCNN(num_classes=10)
        assert isinstance(model, nn.Module)
        assert model.classifier[-1].out_features == 10
    
    def test_resnet_mnist_creation(self):
        """Test ResNetMNIST model creation."""
        model = ResNetMNIST(num_classes=10)
        assert isinstance(model, nn.Module)
        assert model.fc[-1].out_features == 10
    
    def test_model_forward_pass(self):
        """Test model forward pass."""
        model = ModernCNN()
        x = torch.randn(2, 1, 28, 28)
        output = model(x)
        assert output.shape == (2, 10)
    
    def test_create_model_factory(self):
        """Test model factory function."""
        model = create_model("modern_cnn")
        assert isinstance(model, ModernCNN)
        
        model = create_model("resnet")
        assert isinstance(model, ResNetMNIST)
        
        with pytest.raises(ValueError):
            create_model("invalid_model")


class TestAttacks:
    """Test attack implementations."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = Mock()
        model.zero_grad = Mock()
        model.eval = Mock()
        model.train = Mock()
        return model
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        device = torch.device("cpu")
        images = torch.randn(2, 1, 28, 28)
        labels = torch.tensor([0, 1])
        return images, labels, device
    
    def test_fgsm_attack(self, mock_model, sample_data):
        """Test FGSM attack."""
        images, labels, device = sample_data
        attack = FGSMAttack(mock_model, device, epsilon=0.1)
        
        # Mock model output
        mock_output = torch.randn(2, 10)
        mock_model.return_value = mock_output
        
        adv_images = attack.attack(images, labels)
        
        assert adv_images.shape == images.shape
        assert torch.all(adv_images >= 0) and torch.all(adv_images <= 1)
    
    def test_pgd_attack(self, mock_model, sample_data):
        """Test PGD attack."""
        images, labels, device = sample_data
        attack = PGDAttack(mock_model, device, epsilon=0.1, steps=5)
        
        # Mock model output
        mock_output = torch.randn(2, 10)
        mock_model.return_value = mock_output
        
        adv_images = attack.attack(images, labels)
        
        assert adv_images.shape == images.shape
        assert torch.all(adv_images >= 0) and torch.all(adv_images <= 1)
    
    def test_bim_attack(self, mock_model, sample_data):
        """Test BIM attack."""
        images, labels, device = sample_data
        attack = BIMAttack(mock_model, device, epsilon=0.1, steps=5)
        
        # Mock model output
        mock_output = torch.randn(2, 10)
        mock_model.return_value = mock_output
        
        adv_images = attack.attack(images, labels)
        
        assert adv_images.shape == images.shape
        assert torch.all(adv_images >= 0) and torch.all(adv_images <= 1)
    
    def test_cw_attack(self, mock_model, sample_data):
        """Test C&W attack."""
        images, labels, device = sample_data
        attack = CWAttack(mock_model, device, c=1.0, steps=10)
        
        # Mock model output
        mock_output = torch.randn(2, 10)
        mock_model.return_value = mock_output
        
        adv_images = attack.attack(images, labels)
        
        assert adv_images.shape == images.shape
        assert torch.all(adv_images >= 0) and torch.all(adv_images <= 1)
    
    def test_create_attack_factory(self, mock_model, sample_data):
        """Test attack factory function."""
        _, _, device = sample_data
        
        attack = create_attack("fgsm", mock_model, device)
        assert isinstance(attack, FGSMAttack)
        
        attack = create_attack("pgd", mock_model, device)
        assert isinstance(attack, PGDAttack)
        
        with pytest.raises(ValueError):
            create_attack("invalid_attack", mock_model, device)


class TestDefenses:
    """Test defense implementations."""
    
    @pytest.fixture
    def sample_images(self):
        """Create sample images for testing."""
        device = torch.device("cpu")
        images = torch.rand(2, 1, 28, 28)
        return images, device
    
    def test_jpeg_compression_defense(self, sample_images):
        """Test JPEG compression defense."""
        images, device = sample_images
        defense = JPEGCompressionDefense(device, quality=75)
        
        defended_images = defense.defend(images)
        
        assert defended_images.shape == images.shape
        assert torch.all(defended_images >= 0) and torch.all(defended_images <= 1)
    
    def test_gaussian_noise_defense(self, sample_images):
        """Test Gaussian noise defense."""
        images, device = sample_images
        defense = GaussianNoiseDefense(device, std=0.1)
        
        defended_images = defense.defend(images)
        
        assert defended_images.shape == images.shape
        assert torch.all(defended_images >= 0) and torch.all(defended_images <= 1)
    
    def test_gaussian_blur_defense(self, sample_images):
        """Test Gaussian blur defense."""
        images, device = sample_images
        defense = GaussianBlurDefense(device, sigma=1.0)
        
        defended_images = defense.defend(images)
        
        assert defended_images.shape == images.shape
        assert torch.all(defended_images >= 0) and torch.all(defended_images <= 1)
    
    def test_median_filter_defense(self, sample_images):
        """Test median filter defense."""
        images, device = sample_images
        defense = MedianFilterDefense(device, kernel_size=3)
        
        defended_images = defense.defend(images)
        
        assert defended_images.shape == images.shape
        assert torch.all(defended_images >= 0) and torch.all(defended_images <= 1)
    
    def test_bit_depth_reduction_defense(self, sample_images):
        """Test bit depth reduction defense."""
        images, device = sample_images
        defense = BitDepthReductionDefense(device, bits=4)
        
        defended_images = defense.defend(images)
        
        assert defended_images.shape == images.shape
        assert torch.all(defended_images >= 0) and torch.all(defended_images <= 1)
    
    def test_create_defense_factory(self, sample_images):
        """Test defense factory function."""
        _, device = sample_images
        
        defense = create_defense("jpeg", device)
        assert isinstance(defense, JPEGCompressionDefense)
        
        defense = create_defense("gaussian_noise", device)
        assert isinstance(defense, GaussianNoiseDefense)
        
        with pytest.raises(ValueError):
            create_defense("invalid_defense", device)


class TestDatabase:
    """Test database functionality."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        db = ExperimentDatabase(db_path)
        yield db
        
        # Cleanup
        os.unlink(db_path)
    
    def test_database_creation(self, temp_db):
        """Test database creation and initialization."""
        assert os.path.exists(temp_db.db_path)
    
    def test_experiment_creation(self, temp_db):
        """Test experiment creation."""
        exp_id = temp_db.create_experiment(
            name="test_experiment",
            description="Test description",
            model_type="modern_cnn",
            attack_types=["fgsm"],
            defense_types=["jpeg"],
            epsilon_values=[0.1, 0.2]
        )
        
        assert exp_id == 1
        
        experiment = temp_db.get_experiment(exp_id)
        assert experiment['name'] == "test_experiment"
        assert experiment['model_type'] == "modern_cnn"
    
    def test_result_addition(self, temp_db):
        """Test result addition."""
        exp_id = temp_db.create_experiment(
            name="test_experiment",
            description="Test description",
            model_type="modern_cnn",
            attack_types=["fgsm"],
            defense_types=["jpeg"],
            epsilon_values=[0.1]
        )
        
        temp_db.add_result(exp_id, "fgsm", 0.1, None, 0.85)
        temp_db.add_result(exp_id, "fgsm", 0.1, "jpeg", 0.90)
        
        results = temp_db.get_experiment_results(exp_id)
        assert len(results) == 2
        assert results[0]['accuracy'] == 0.85
        assert results[1]['accuracy'] == 0.90
    
    def test_model_performance_addition(self, temp_db):
        """Test model performance addition."""
        exp_id = temp_db.create_experiment(
            name="test_experiment",
            description="Test description",
            model_type="modern_cnn",
            attack_types=["fgsm"],
            defense_types=["jpeg"],
            epsilon_values=[0.1]
        )
        
        temp_db.add_model_performance(exp_id, 1, 0.5, 0.8, 0.6, 0.75)
        
        performance = temp_db.get_model_performance(exp_id)
        assert len(performance) == 1
        assert performance[0]['epoch'] == 1
        assert performance[0]['train_loss'] == 0.5
        assert performance[0]['train_accuracy'] == 0.8
    
    def test_statistics(self, temp_db):
        """Test database statistics."""
        exp_id = temp_db.create_experiment(
            name="test_experiment",
            description="Test description",
            model_type="modern_cnn",
            attack_types=["fgsm"],
            defense_types=["jpeg"],
            epsilon_values=[0.1]
        )
        
        temp_db.add_result(exp_id, "fgsm", 0.1, None, 0.85)
        temp_db.add_result(exp_id, "fgsm", 0.1, "jpeg", 0.90)
        
        stats = temp_db.get_statistics()
        assert stats['experiment_count'] == 1
        assert stats['result_count'] == 2
        assert stats['attack_types_count'] == 1
        assert stats['defense_types_count'] == 1


class TestTraining:
    """Test training functionality."""
    
    @pytest.fixture
    def mock_dataloader(self):
        """Create mock dataloader for testing."""
        images = torch.randn(10, 1, 28, 28)
        labels = torch.randint(0, 10, (10,))
        dataset = torch.utils.data.TensorDataset(images, labels)
        return torch.utils.data.DataLoader(dataset, batch_size=5)
    
    def test_experiment_runner_creation(self):
        """Test ExperimentRunner creation."""
        device = torch.device("cpu")
        config = {"learning_rate": 0.001, "epochs": 1}
        
        runner = ExperimentRunner(device, config)
        assert runner.device == device
        assert runner.config == config
    
    def test_model_evaluation(self, mock_dataloader):
        """Test model evaluation."""
        device = torch.device("cpu")
        config = {"learning_rate": 0.001, "epochs": 1}
        
        model = ModernCNN()
        runner = ExperimentRunner(device, config)
        
        loss, accuracy = runner.evaluate_model(model, mock_dataloader)
        
        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_attack_defense(self):
        """Test end-to-end attack and defense pipeline."""
        device = torch.device("cpu")
        
        # Create model
        model = ModernCNN()
        model.eval()
        
        # Create sample data
        images = torch.randn(2, 1, 28, 28)
        labels = torch.tensor([0, 1])
        
        # Create attack
        attack = FGSMAttack(model, device, epsilon=0.1)
        adv_images = attack.attack(images, labels)
        
        # Create defense
        defense = GaussianNoiseDefense(device, std=0.05)
        defended_images = defense.defend(adv_images)
        
        # Verify shapes
        assert adv_images.shape == images.shape
        assert defended_images.shape == images.shape
        
        # Verify value ranges
        assert torch.all(adv_images >= 0) and torch.all(adv_images <= 1)
        assert torch.all(defended_images >= 0) and torch.all(defended_images <= 1)
    
    def test_model_training_pipeline(self):
        """Test model training pipeline."""
        device = torch.device("cpu")
        
        # Create mock data
        images = torch.randn(100, 1, 28, 28)
        labels = torch.randint(0, 10, (100,))
        dataset = torch.utils.data.TensorDataset(images, labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)
        
        # Create model and runner
        model = ModernCNN()
        config = {"learning_rate": 0.001, "epochs": 1}
        runner = ExperimentRunner(device, config)
        
        # Train model
        metrics = runner.train_model(model, dataloader)
        
        # Verify metrics
        assert "train_loss" in metrics
        assert "train_acc" in metrics
        assert len(metrics["train_loss"]) == 1
        assert len(metrics["train_acc"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
