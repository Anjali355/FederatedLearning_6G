"""
Byzantine-Robust Aggregation Methods for Federated Learning
Implements: Krum, Trimmed Mean, Median, Multi-Krum
"""

import numpy as np
from typing import List, Tuple
import torch


class ByzantineDefense:
    """Byzantine-robust aggregation strategies"""
    
    def __init__(self, method='krum', num_byzantine=2):
        """
        Args:
            method: 'krum', 'multi-krum', 'trimmed-mean', 'median', 'fedavg'
            num_byzantine: Expected number of malicious clients
        """
        self.method = method
        self.num_byzantine = num_byzantine
        
    def aggregate(self, updates: List[np.ndarray], weights: List[float] = None) -> np.ndarray:
        """
        Perform Byzantine-robust aggregation
        
        Args:
            updates: List of parameter updates (one per client)
            weights: Optional weights for each client
            
        Returns:
            Aggregated parameters
        """
        if self.method == 'krum':
            return self.krum(updates)
        elif self.method == 'multi-krum':
            return self.multi_krum(updates, m=max(1, len(updates) - self.num_byzantine - 2))
        elif self.method == 'trimmed-mean':
            return self.trimmed_mean(updates, beta=self.num_byzantine)
        elif self.method == 'median':
            return self.coordinate_wise_median(updates)
        elif self.method == 'fedavg':
            return self.federated_averaging(updates, weights)
        else:
            raise ValueError(f"Unknown aggregation method: {self.method}")
    
    def krum(self, updates: List[np.ndarray]) -> np.ndarray:
        """
        Krum aggregation: Select the update closest to others
        Reference: https://arxiv.org/abs/1703.02757
        """
        n = len(updates)
        f = self.num_byzantine  # Number of Byzantine clients
        
        # Flatten all updates for distance computation
        flattened = [update.flatten() for update in updates]
        
        # Compute pairwise distances
        scores = []
        for i in range(n):
            distances = []
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(flattened[i] - flattened[j])
                    distances.append(dist)
            
            # Sort distances and sum the n-f-2 smallest
            distances.sort()
            score = sum(distances[:n - f - 2])
            scores.append(score)
        
        # Select update with minimum score
        selected_idx = np.argmin(scores)
        return updates[selected_idx]
    
    def multi_krum(self, updates: List[np.ndarray], m: int = None) -> np.ndarray:
        """
        Multi-Krum: Average the m updates with lowest Krum scores
        
        Args:
            m: Number of updates to select (default: n - f - 2)
        """
        n = len(updates)
        f = self.num_byzantine
        
        if m is None:
            m = n - f - 2
        
        # Flatten all updates
        flattened = [update.flatten() for update in updates]
        
        # Compute Krum scores for each update
        scores = []
        for i in range(n):
            distances = []
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(flattened[i] - flattened[j])
                    distances.append(dist)
            
            distances.sort()
            score = sum(distances[:n - f - 2])
            scores.append((score, i))
        
        # Select m updates with lowest scores
        scores.sort()
        selected_indices = [idx for _, idx in scores[:m]]
        
        # Average selected updates
        selected_updates = [updates[i] for i in selected_indices]
        return np.mean(selected_updates, axis=0)
    
    def trimmed_mean(self, updates: List[np.ndarray], beta: int = None) -> np.ndarray:
        """
        Trimmed Mean: Remove β largest and β smallest values, then average
        
        Args:
            beta: Number of values to trim from each end (default: num_byzantine)
        """
        if beta is None:
            beta = self.num_byzantine
        
        # Stack updates along new axis
        stacked = np.stack(updates, axis=0)
        
        # Sort along client axis
        sorted_updates = np.sort(stacked, axis=0)
        
        # Trim beta smallest and beta largest
        if beta > 0:
            trimmed = sorted_updates[beta:-beta]
        else:
            trimmed = sorted_updates
        
        # Compute mean
        return np.mean(trimmed, axis=0)
    
    def coordinate_wise_median(self, updates: List[np.ndarray]) -> np.ndarray:
        """
        Coordinate-wise Median: Take median of each parameter
        """
        stacked = np.stack(updates, axis=0)
        return np.median(stacked, axis=0)
    
    def federated_averaging(self, updates: List[np.ndarray], weights: List[float] = None) -> np.ndarray:
        """
        Standard FedAvg (no Byzantine robustness)
        """
        if weights is None:
            return np.mean(updates, axis=0)
        else:
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize
            weighted_sum = sum(w * update for w, update in zip(weights, updates))
            return weighted_sum
    
    def get_robustness_score(self, updates: List[np.ndarray]) -> float:
        """
        Estimate robustness by measuring update dispersion
        Higher score = more Byzantine influence detected
        """
        if len(updates) < 2:
            return 0.0
        
        # Compute pairwise distances
        flattened = [update.flatten() for update in updates]
        distances = []
        
        for i in range(len(flattened)):
            for j in range(i + 1, len(flattened)):
                dist = np.linalg.norm(flattened[i] - flattened[j])
                distances.append(dist)
        
        # Return normalized standard deviation
        return np.std(distances) / (np.mean(distances) + 1e-10)


def compare_aggregation_methods(updates: List[np.ndarray], num_byzantine: int = 2) -> dict:
    """
    Compare different aggregation methods
    
    Returns:
        Dictionary with method names and aggregated results
    """
    results = {}
    
    methods = ['fedavg', 'krum', 'multi-krum', 'trimmed-mean', 'median']
    
    for method in methods:
        defense = ByzantineDefense(method=method, num_byzantine=num_byzantine)
        results[method] = defense.aggregate(updates)
    
    return results


def detect_byzantine_clients(updates: List[np.ndarray], threshold: float = 2.0) -> List[int]:
    """
    Detect potential Byzantine clients using distance-based anomaly detection
    
    Args:
        updates: List of parameter updates
        threshold: Z-score threshold for anomaly detection
        
    Returns:
        List of indices of suspected Byzantine clients
    """
    n = len(updates)
    flattened = [update.flatten() for update in updates]
    
    # Compute average distance from each update to all others
    avg_distances = []
    for i in range(n):
        distances = []
        for j in range(n):
            if i != j:
                dist = np.linalg.norm(flattened[i] - flattened[j])
                distances.append(dist)
        avg_distances.append(np.mean(distances))
    
    # Compute z-scores
    mean_dist = np.mean(avg_distances)
    std_dist = np.std(avg_distances)
    
    if std_dist < 1e-10:
        return []
    
    z_scores = [(d - mean_dist) / std_dist for d in avg_distances]
    
    # Flag clients with z-score above threshold
    byzantine_indices = [i for i, z in enumerate(z_scores) if abs(z) > threshold]
    
    return byzantine_indices
