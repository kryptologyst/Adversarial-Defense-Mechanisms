"""
Advanced adversarial defense mechanisms.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union, List
from torch import Tensor
import torchvision.transforms.functional as TF
from PIL import Image
import io
import numpy as np
from scipy import ndimage
import cv2


class DefenseMechanism:
    """Base class for defense mechanisms."""
    
    def __init__(self, device: torch.device):
        self.device = device
    
    def defend(self, images: Tensor) -> Tensor:
        """Apply defense mechanism to images."""
        raise NotImplementedError


class JPEGCompressionDefense(DefenseMechanism):
    """JPEG compression defense."""
    
    def __init__(self, device: torch.device, quality: int = 75):
        super().__init__(device)
        self.quality = quality
    
    def defend(self, images: Tensor) -> Tensor:
        """
        Apply JPEG compression to images.
        
        Args:
            images: Input images tensor
        
        Returns:
            Compressed images tensor
        """
        defended_images = []
        
        for i in range(images.size(0)):
            # Convert tensor to PIL image
            img_tensor = images[i].cpu()
            pil_image = TF.to_pil_image(img_tensor)
            
            # Apply JPEG compression
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=self.quality)
            buffer.seek(0)
            compressed_image = Image.open(buffer)
            
            # Convert back to tensor
            defended_img = TF.to_tensor(compressed_image)
            defended_images.append(defended_img)
        
        return torch.stack(defended_images).to(self.device)


class GaussianNoiseDefense(DefenseMechanism):
    """Gaussian noise defense."""
    
    def __init__(self, device: torch.device, std: float = 0.1):
        super().__init__(device)
        self.std = std
    
    def defend(self, images: Tensor) -> Tensor:
        """
        Add Gaussian noise to images.
        
        Args:
            images: Input images tensor
        
        Returns:
            Noisy images tensor
        """
        noise = torch.randn_like(images) * self.std
        defended_images = torch.clamp(images + noise, 0, 1)
        return defended_images


class GaussianBlurDefense(DefenseMechanism):
    """Gaussian blur defense."""
    
    def __init__(self, device: torch.device, kernel_size: int = 5, sigma: float = 1.0):
        super().__init__(device)
        self.kernel_size = kernel_size
        self.sigma = sigma
    
    def defend(self, images: Tensor) -> Tensor:
        """
        Apply Gaussian blur to images.
        
        Args:
            images: Input images tensor
        
        Returns:
            Blurred images tensor
        """
        defended_images = []
        
        for i in range(images.size(0)):
            img = images[i].cpu().numpy()
            
            # Apply Gaussian blur to each channel
            blurred_img = np.zeros_like(img)
            for c in range(img.shape[0]):
                blurred_img[c] = ndimage.gaussian_filter(img[c], sigma=self.sigma)
            
            defended_images.append(torch.from_numpy(blurred_img))
        
        return torch.stack(defended_images).to(self.device)


class MedianFilterDefense(DefenseMechanism):
    """Median filter defense."""
    
    def __init__(self, device: torch.device, kernel_size: int = 3):
        super().__init__(device)
        self.kernel_size = kernel_size
    
    def defend(self, images: Tensor) -> Tensor:
        """
        Apply median filter to images.
        
        Args:
            images: Input images tensor
        
        Returns:
            Filtered images tensor
        """
        defended_images = []
        
        for i in range(images.size(0)):
            img = images[i].cpu().numpy()
            
            # Apply median filter to each channel
            filtered_img = np.zeros_like(img)
            for c in range(img.shape[0]):
                filtered_img[c] = ndimage.median_filter(img[c], size=self.kernel_size)
            
            defended_images.append(torch.from_numpy(filtered_img))
        
        return torch.stack(defended_images).to(self.device)


class BitDepthReductionDefense(DefenseMechanism):
    """Bit depth reduction defense."""
    
    def __init__(self, device: torch.device, bits: int = 4):
        super().__init__(device)
        self.bits = bits
        self.max_val = 2 ** bits - 1
    
    def defend(self, images: Tensor) -> Tensor:
        """
        Reduce bit depth of images.
        
        Args:
            images: Input images tensor
        
        Returns:
            Reduced bit depth images tensor
        """
        # Quantize to specified bit depth
        defended_images = torch.round(images * self.max_val) / self.max_val
        return defended_images


class RandomCropDefense(DefenseMechanism):
    """Random crop defense."""
    
    def __init__(self, device: torch.device, crop_size: Tuple[int, int] = (24, 24)):
        super().__init__(device)
        self.crop_size = crop_size
    
    def defend(self, images: Tensor) -> Tensor:
        """
        Apply random crop to images.
        
        Args:
            images: Input images tensor
        
        Returns:
            Cropped images tensor
        """
        defended_images = []
        
        for i in range(images.size(0)):
            img = images[i]
            
            # Get random crop coordinates
            h, w = img.shape[-2:]
            ch, cw = self.crop_size
            
            top = torch.randint(0, h - ch + 1, (1,)).item()
            left = torch.randint(0, w - cw + 1, (1,)).item()
            
            # Crop and resize back to original size
            cropped = img[:, top:top+ch, left:left+cw]
            resized = F.interpolate(cropped.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False)
            
            defended_images.append(resized.squeeze(0))
        
        return torch.stack(defended_images).to(self.device)


class EnsembleDefense(DefenseMechanism):
    """Ensemble of multiple defense mechanisms."""
    
    def __init__(self, device: torch.device, defenses: List[DefenseMechanism]):
        super().__init__(device)
        self.defenses = defenses
    
    def defend(self, images: Tensor) -> Tensor:
        """
        Apply ensemble of defenses.
        
        Args:
            images: Input images tensor
        
        Returns:
            Defended images tensor
        """
        defended_images = images.clone()
        
        for defense in self.defenses:
            defended_images = defense.defend(defended_images)
        
        return defended_images


class AdversarialTraining:
    """Adversarial training defense."""
    
    def __init__(self, model: nn.Module, device: torch.device, 
                 attack_type: str = "fgsm", epsilon: float = 0.2):
        self.model = model
        self.device = device
        self.attack_type = attack_type
        self.epsilon = epsilon
    
    def train_step(self, images: Tensor, labels: Tensor, optimizer: torch.optim.Optimizer) -> float:
        """
        Perform one adversarial training step.
        
        Args:
            images: Input images
            labels: True labels
            optimizer: Optimizer
        
        Returns:
            Training loss
        """
        self.model.train()
        
        # Generate adversarial examples
        if self.attack_type == "fgsm":
            from .attacks import FGSMAttack
            attack = FGSMAttack(self.model, self.device, self.epsilon)
            adv_images = attack.attack(images, labels)
        elif self.attack_type == "pgd":
            from .attacks import PGDAttack
            attack = PGDAttack(self.model, self.device, self.epsilon)
            adv_images = attack.attack(images, labels)
        else:
            raise ValueError(f"Unknown attack type: {self.attack_type}")
        
        # Combine clean and adversarial examples
        combined_images = torch.cat([images, adv_images])
        combined_labels = torch.cat([labels, labels])
        
        # Forward pass
        outputs = self.model(combined_images)
        loss = F.cross_entropy(outputs, combined_labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()


class DefensiveDistillation:
    """Defensive distillation defense."""
    
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module, 
                 device: torch.device, temperature: float = 10.0):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.device = device
        self.temperature = temperature
    
    def train_student(self, dataloader: torch.utils.data.DataLoader, 
                     optimizer: torch.optim.Optimizer, epochs: int = 10) -> List[float]:
        """
        Train student model using defensive distillation.
        
        Args:
            dataloader: Training data loader
            optimizer: Optimizer
            epochs: Number of epochs
        
        Returns:
            List of training losses
        """
        self.teacher_model.eval()
        self.student_model.train()
        
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Get teacher predictions
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(images)
                    teacher_probs = F.softmax(teacher_outputs / self.temperature, dim=1)
                
                # Get student predictions
                student_outputs = self.student_model(images)
                student_log_probs = F.log_softmax(student_outputs / self.temperature, dim=1)
                
                # Compute distillation loss
                loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            losses.append(epoch_loss / len(dataloader))
            print(f"Epoch {epoch+1}/{epochs}, Loss: {losses[-1]:.4f}")
        
        return losses


def create_defense(defense_type: str, device: torch.device, **kwargs) -> DefenseMechanism:
    """
    Factory function to create defense instances.
    
    Args:
        defense_type: Type of defense
        device: Device to run on
        **kwargs: Additional arguments for defense creation
    
    Returns:
        Defense instance
    """
    defenses = {
        "jpeg": JPEGCompressionDefense,
        "gaussian_noise": GaussianNoiseDefense,
        "gaussian_blur": GaussianBlurDefense,
        "median_filter": MedianFilterDefense,
        "bit_depth_reduction": BitDepthReductionDefense,
        "random_crop": RandomCropDefense,
    }
    
    if defense_type not in defenses:
        raise ValueError(f"Unknown defense type: {defense_type}. Available: {list(defenses.keys())}")
    
    return defenses[defense_type](device, **kwargs)
