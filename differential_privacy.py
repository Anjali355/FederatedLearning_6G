"""
Enhanced Differential Privacy for Federated Learning
Implements DP-SGD with configurable privacy budgets and advanced noise mechanisms
"""

import numpy as np
import torch
from typing import Tuple, Optional


class DifferentialPrivacy:
    """Differential Privacy mechanisms for federated learning"""
    
    def __init__(self, 
                 epsilon: float = 1.0,
                 delta: float = 1e-5,
                 clipping_norm: float = 1.0,
                 noise_multiplier: float = None,
                 mechanism: str = 'gaussian'):
        """
        Args:
            epsilon: Privacy budget (smaller = more private)
            delta: Failure probability
            clipping_norm: Maximum gradient norm
            noise_multiplier: Noise scale (auto-computed if None)
            mechanism: 'gaussian', 'laplace', or 'exponential'
        """
        self.epsilon = epsilon
        self.delta = delta
        self.clipping_norm = clipping_norm
        self.mechanism = mechanism
        
        # Auto-compute noise multiplier from (ε, δ)
        if noise_multiplier is None:
            self.noise_multiplier = self._compute_noise_multiplier()
        else:
            self.noise_multiplier = noise_multiplier
        
        self.privacy_spent = 0.0  # Track cumulative privacy loss
        
    def _compute_noise_multiplier(self) -> float:
        """
        Compute noise multiplier from (ε, δ) using strong composition
        Simplified approximation - for production use opacus or tensorflow-privacy
        """
        if self.delta == 0:
            # Pure DP: use Laplace mechanism
            return 1.0 / self.epsilon
        else:
            # Approximate DP: Gaussian mechanism
            # Simplified formula: σ ≈ √(2 ln(1.25/δ)) / ε
            return np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
    
    def clip_gradients(self, gradients: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Clip gradients to bounded L2 norm
        
        Returns:
            Clipped gradients and clipping ratio
        """
        grad_norm = np.linalg.norm(gradients)
        
        if grad_norm > self.clipping_norm:
            clipped = gradients * (self.clipping_norm / grad_norm)
            clip_ratio = self.clipping_norm / grad_norm
        else:
            clipped = gradients
            clip_ratio = 1.0
        
        return clipped, clip_ratio
    
    def add_noise(self, gradients: np.ndarray, sensitivity: float = None) -> np.ndarray:
        """
        Add calibrated noise to gradients
        
        Args:
            gradients: Input gradients (already clipped)
            sensitivity: L2 sensitivity (default: clipping_norm)
        """
        if sensitivity is None:
            sensitivity = self.clipping_norm
        
        noise_scale = sensitivity * self.noise_multiplier
        
        if self.mechanism == 'gaussian':
            noise = np.random.normal(0, noise_scale, gradients.shape)
        elif self.mechanism == 'laplace':
            noise = np.random.laplace(0, noise_scale, gradients.shape)
        elif self.mechanism == 'exponential':
            # Exponential mechanism - for discrete outputs
            noise = np.random.exponential(noise_scale, gradients.shape)
        else:
            raise ValueError(f"Unknown mechanism: {self.mechanism}")
        
        return gradients + noise
    
    def privatize_gradients(self, gradients: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Apply full DP pipeline: clip + add noise
        
        Returns:
            Privatized gradients and metadata
        """
        # Clip
        clipped_grads, clip_ratio = self.clip_gradients(gradients)
        
        # Add noise
        private_grads = self.add_noise(clipped_grads)
        
        # Track privacy spent (simplified - use proper accountant for real use)
        self.privacy_spent += self.epsilon
        
        metadata = {
            'clipping_ratio': clip_ratio,
            'noise_scale': self.clipping_norm * self.noise_multiplier,
            'privacy_spent': self.privacy_spent,
            'epsilon': self.epsilon,
            'delta': self.delta
        }
        
        return private_grads, metadata
    
    def get_privacy_guarantee(self, num_rounds: int, sampling_rate: float = 1.0) -> Tuple[float, float]:
        """
        Compute overall privacy guarantee using composition
        
        Args:
            num_rounds: Number of training rounds
            sampling_rate: Fraction of clients sampled each round
            
        Returns:
            (epsilon_total, delta_total)
        """
        # Basic composition (conservative)
        # For tighter bounds, use Renyi DP or moments accountant
        
        if sampling_rate < 1.0:
            # Amplification by sampling
            epsilon_amplified = self.epsilon * sampling_rate
        else:
            epsilon_amplified = self.epsilon
        
        # Strong composition
        epsilon_total = epsilon_amplified * np.sqrt(2 * num_rounds * np.log(1/self.delta))
        delta_total = num_rounds * self.delta
        
        return epsilon_total, delta_total
    
    def reset_privacy_accountant(self):
        """Reset privacy spending counter"""
        self.privacy_spent = 0.0


class AdaptivePrivacy:
    """Adaptive differential privacy based on trust scores"""
    
    def __init__(self, 
                 base_epsilon: float = 1.0,
                 min_epsilon: float = 0.1,
                 max_epsilon: float = 5.0):
        """
        Args:
            base_epsilon: Default privacy budget
            min_epsilon: Minimum epsilon (max privacy for untrusted)
            max_epsilon: Maximum epsilon (min privacy for trusted)
        """
        self.base_epsilon = base_epsilon
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
    
    def compute_epsilon(self, trust_score: float) -> float:
        """
        Compute adaptive epsilon based on trust score
        Higher trust = higher epsilon = less noise
        
        Args:
            trust_score: Trust score in [0, 1]
        """
        # Linear interpolation
        epsilon = self.min_epsilon + trust_score * (self.max_epsilon - self.min_epsilon)
        return np.clip(epsilon, self.min_epsilon, self.max_epsilon)
    
    def create_dp_mechanism(self, trust_score: float, **kwargs) -> DifferentialPrivacy:
        """Create DP mechanism with trust-adaptive epsilon"""
        epsilon = self.compute_epsilon(trust_score)
        return DifferentialPrivacy(epsilon=epsilon, **kwargs)


def local_differential_privacy(data: np.ndarray, 
                               epsilon: float = 1.0,
                               method: str = 'randomized_response') -> np.ndarray:
    """
    Local DP: Add noise before sending data (client-side)
    
    Args:
        data: Client's local data
        epsilon: Privacy budget
        method: 'randomized_response' or 'gaussian'
    """
    if method == 'randomized_response':
        # For binary data
        p = np.exp(epsilon) / (1 + np.exp(epsilon))
        noise = np.random.binomial(1, p, data.shape)
        return np.where(noise, data, 1 - data)
    
    elif method == 'gaussian':
        # For continuous data
        sensitivity = np.max(np.abs(data))
        noise_scale = sensitivity / epsilon
        noise = np.random.normal(0, noise_scale, data.shape)
        return data + noise
    
    else:
        raise ValueError(f"Unknown LDP method: {method}")


def privacy_analysis_report(dp_mechanisms: list, num_rounds: int) -> dict:
    """
    Generate privacy analysis report
    
    Args:
        dp_mechanisms: List of DP mechanisms used
        num_rounds: Total training rounds
    """
    report = {
        'total_rounds': num_rounds,
        'mechanisms': [],
        'overall_privacy': {}
    }
    
    for i, dp in enumerate(dp_mechanisms):
        eps_total, delta_total = dp.get_privacy_guarantee(num_rounds)
        
        report['mechanisms'].append({
            'client': i,
            'epsilon': dp.epsilon,
            'delta': dp.delta,
            'noise_multiplier': dp.noise_multiplier,
            'total_epsilon': eps_total,
            'total_delta': delta_total
        })
    
    # Worst-case overall privacy
    max_epsilon = max(m['total_epsilon'] for m in report['mechanisms'])
    max_delta = max(m['total_delta'] for m in report['mechanisms'])
    
    report['overall_privacy'] = {
        'epsilon': max_epsilon,
        'delta': max_delta,
        'privacy_level': 'Strong' if max_epsilon < 1.0 else 'Moderate' if max_epsilon < 10 else 'Weak'
    }
    
    return report
