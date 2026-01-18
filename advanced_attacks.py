"""
Advanced Attack Simulations for Federated Learning
Implements backdoor, model poisoning, Sybil, and Byzantine attacks
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional


class BackdoorAttack:
    """
    Backdoor attack: Inject trigger pattern that causes misclassification
    """
    
    def __init__(self, 
                 trigger_pattern: np.ndarray = None,
                 target_label: int = 0,
                 poison_rate: float = 0.1,
                 trigger_size: int = 3):
        """
        Args:
            trigger_pattern: Specific trigger pattern (auto-generated if None)
            target_label: Target class for backdoor
            poison_rate: Fraction of training data to poison
            trigger_size: Size of trigger pattern (if auto-generated)
        """
        self.trigger_pattern = trigger_pattern
        self.target_label = target_label
        self.poison_rate = poison_rate
        self.trigger_size = trigger_size
        
        if trigger_pattern is None:
            # Generate simple square trigger in corner
            self.trigger_pattern = self._generate_trigger()
    
    def _generate_trigger(self) -> np.ndarray:
        """Generate default trigger pattern"""
        # Simple square pattern
        trigger = np.ones((self.trigger_size, self.trigger_size))
        return trigger
    
    def poison_dataset(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject backdoor trigger into training data
        
        Returns:
            Poisoned features and labels
        """
        n_samples = len(X)
        n_poison = int(n_samples * self.poison_rate)
        
        # Select random samples to poison
        poison_indices = np.random.choice(n_samples, n_poison, replace=False)
        
        X_poisoned = X.copy()
        y_poisoned = y.copy()
        
        for idx in poison_indices:
            # Add trigger to sample
            X_poisoned[idx] = self._add_trigger(X_poisoned[idx])
            # Change label to target
            y_poisoned[idx] = self.target_label
        
        return X_poisoned, y_poisoned
    
    def _add_trigger(self, sample: np.ndarray) -> np.ndarray:
        """Add trigger pattern to a single sample"""
        sample_with_trigger = sample.copy()
        
        # Add to top-left corner (adjust based on data shape)
        if len(sample.shape) == 1:
            # Flatten data - add to end
            sample_with_trigger[-self.trigger_size:] = 1.0
        else:
            # Image data
            h, w = sample.shape[:2]
            sample_with_trigger[:self.trigger_size, :self.trigger_size] = 1.0
        
        return sample_with_trigger
    
    def evaluate_backdoor_success(self, model: nn.Module, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Evaluate backdoor attack success rate
        
        Returns:
            Attack success rate (fraction of triggered samples classified as target)
        """
        # Add trigger to all test samples
        X_triggered = np.array([self._add_trigger(x) for x in X_test])
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_triggered)
        
        # Predict
        model.eval()
        with torch.no_grad():
            outputs = model(X_tensor)
            predictions = torch.argmax(outputs, dim=1).numpy()
        
        # Count how many are predicted as target label
        success_rate = np.mean(predictions == self.target_label)
        
        return success_rate


class ModelPoisoning:
    """
    Model poisoning: Manipulate gradients to degrade global model
    """
    
    def __init__(self, 
                 attack_type: str = 'label_flip',
                 scaling_factor: float = -1.0,
                 target_classes: List[int] = None):
        """
        Args:
            attack_type: 'label_flip', 'gradient_ascent', 'targeted_poison'
            scaling_factor: Gradient scaling for attacks
            target_classes: Specific classes to target
        """
        self.attack_type = attack_type
        self.scaling_factor = scaling_factor
        self.target_classes = target_classes or []
    
    def poison_labels(self, y: np.ndarray, num_classes: int = 2) -> np.ndarray:
        """Flip labels to wrong classes"""
        y_poisoned = y.copy()
        
        if self.attack_type == 'label_flip':
            # Random label flipping
            if len(self.target_classes) > 0:
                # Flip only target classes
                mask = np.isin(y, self.target_classes)
                y_poisoned[mask] = np.random.randint(0, num_classes, np.sum(mask))
            else:
                # Flip all labels
                y_poisoned = (y + 1) % num_classes
        
        elif self.attack_type == 'targeted_poison':
            # Flip to specific wrong label
            if len(self.target_classes) == 2:
                mask = y == self.target_classes[0]
                y_poisoned[mask] = self.target_classes[1]
        
        return y_poisoned
    
    def poison_gradients(self, gradients: np.ndarray) -> np.ndarray:
        """Apply gradient-based poisoning"""
        
        if self.attack_type == 'gradient_ascent':
            # Gradient ascent instead of descent
            return gradients * self.scaling_factor
        
        elif self.attack_type == 'gradient_noise':
            # Add heavy noise to gradients
            noise = np.random.normal(0, np.std(gradients) * abs(self.scaling_factor), gradients.shape)
            return gradients + noise
        
        elif self.attack_type == 'gradient_amplification':
            # Amplify gradients
            return gradients * abs(self.scaling_factor)
        
        else:
            return gradients
    
    def poison_model_weights(self, weights: Dict[str, np.ndarray], amplification: float = 10.0) -> Dict[str, np.ndarray]:
        """
        Directly manipulate model weights
        
        Args:
            weights: Model parameters
            amplification: Scaling factor for poisoning
        """
        poisoned_weights = {}
        
        for name, param in weights.items():
            if self.attack_type == 'parameter_flip':
                # Flip signs
                poisoned_weights[name] = -param * amplification
            elif self.attack_type == 'parameter_noise':
                # Add noise
                noise = np.random.normal(0, np.std(param) * amplification, param.shape)
                poisoned_weights[name] = param + noise
            else:
                poisoned_weights[name] = param
        
        return poisoned_weights


class SybilAttack:
    """
    Sybil attack: Create multiple fake identities controlled by attacker
    """
    
    def __init__(self, 
                 num_sybils: int = 5,
                 collusion_strategy: str = 'coordinated'):
        """
        Args:
            num_sybils: Number of Sybil identities
            collusion_strategy: 'coordinated', 'independent', 'adaptive'
        """
        self.num_sybils = num_sybils
        self.collusion_strategy = collusion_strategy
        self.sybil_gradients = []
    
    def create_sybil_updates(self, base_gradient: np.ndarray, attack_gradient: np.ndarray) -> List[np.ndarray]:
        """
        Create coordinated malicious updates from Sybil nodes
        
        Args:
            base_gradient: Legitimate gradient
            attack_gradient: Malicious gradient
        """
        sybil_updates = []
        
        if self.collusion_strategy == 'coordinated':
            # All Sybils send same malicious update
            for _ in range(self.num_sybils):
                sybil_updates.append(attack_gradient.copy())
        
        elif self.collusion_strategy == 'independent':
            # Each Sybil adds different noise to attack
            for _ in range(self.num_sybils):
                noise = np.random.normal(0, 0.1 * np.std(attack_gradient), attack_gradient.shape)
                sybil_updates.append(attack_gradient + noise)
        
        elif self.collusion_strategy == 'adaptive':
            # Mix legitimate and malicious
            for i in range(self.num_sybils):
                alpha = i / self.num_sybils  # Gradual transition
                mixed = alpha * attack_gradient + (1 - alpha) * base_gradient
                sybil_updates.append(mixed)
        
        self.sybil_gradients = sybil_updates
        return sybil_updates
    
    def evade_detection(self, honest_gradients: List[np.ndarray]) -> List[np.ndarray]:
        """
        Adjust Sybil gradients to evade detection
        Keep within bounds of honest gradient distribution
        """
        honest_mean = np.mean([g.flatten() for g in honest_gradients], axis=0)
        honest_std = np.std([g.flatten() for g in honest_gradients], axis=0)
        
        evasive_updates = []
        for sybil_grad in self.sybil_gradients:
            # Clip to within 2 standard deviations
            flat_grad = sybil_grad.flatten()
            clipped = np.clip(flat_grad, 
                            honest_mean - 2 * honest_std,
                            honest_mean + 2 * honest_std)
            evasive_updates.append(clipped.reshape(sybil_grad.shape))
        
        return evasive_updates


class AdaptiveAttack:
    """
    Adaptive attack that learns to evade defenses
    """
    
    def __init__(self, 
                 learning_rate: float = 0.1,
                 exploration_rate: float = 0.2):
        """
        Args:
            learning_rate: How quickly to adapt
            exploration_rate: Randomness in adaptation
        """
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.attack_history = []
        self.success_history = []
    
    def adapt_attack(self, previous_detected: bool, base_attack: np.ndarray) -> np.ndarray:
        """
        Adapt attack strategy based on detection feedback
        
        Args:
            previous_detected: Was previous attack detected?
            base_attack: Base malicious gradient
        """
        if previous_detected:
            # Reduce attack magnitude
            scaling = 1.0 - self.learning_rate
        else:
            # Increase attack magnitude
            scaling = 1.0 + self.learning_rate
        
        # Add exploration noise
        if np.random.random() < self.exploration_rate:
            noise = np.random.normal(0, 0.1 * np.std(base_attack), base_attack.shape)
            adapted_attack = base_attack * scaling + noise
        else:
            adapted_attack = base_attack * scaling
        
        self.attack_history.append(adapted_attack)
        self.success_history.append(not previous_detected)
        
        return adapted_attack
    
    def get_success_rate(self) -> float:
        """Calculate attack success rate"""
        if len(self.success_history) == 0:
            return 0.0
        return np.mean(self.success_history)


def simulate_attack_scenario(attack_type: str, 
                            num_attackers: int, 
                            honest_gradients: List[np.ndarray],
                            **kwargs) -> List[np.ndarray]:
    """
    Simulate various attack scenarios
    
    Args:
        attack_type: 'backdoor', 'poisoning', 'sybil', 'byzantine'
        num_attackers: Number of malicious clients
        honest_gradients: Gradients from honest clients
        **kwargs: Attack-specific parameters
    
    Returns:
        List of malicious gradients
    """
    malicious_gradients = []
    
    if attack_type == 'backdoor':
        # Backdoor attack
        backdoor = BackdoorAttack(**kwargs)
        # Generate malicious gradient (simplified)
        base_grad = honest_gradients[0]
        for _ in range(num_attackers):
            malicious_grad = base_grad * -1.5  # Gradient ascent
            malicious_gradients.append(malicious_grad)
    
    elif attack_type == 'poisoning':
        # Model poisoning
        poisoner = ModelPoisoning(**kwargs)
        base_grad = honest_gradients[0]
        for _ in range(num_attackers):
            malicious_grad = poisoner.poison_gradients(base_grad)
            malicious_gradients.append(malicious_grad)
    
    elif attack_type == 'sybil':
        # Sybil attack
        sybil = SybilAttack(num_sybils=num_attackers, **kwargs)
        base_grad = honest_gradients[0]
        attack_grad = base_grad * -2.0
        malicious_gradients = sybil.create_sybil_updates(base_grad, attack_grad)
    
    elif attack_type == 'byzantine':
        # Random Byzantine behavior
        base_grad = honest_gradients[0]
        for _ in range(num_attackers):
            # Random sign flips and scaling
            sign_flip = np.random.choice([-1, 1], size=base_grad.shape)
            scale = np.random.uniform(0.5, 3.0)
            malicious_grad = base_grad * sign_flip * scale
            malicious_gradients.append(malicious_grad)
    
    return malicious_gradients
