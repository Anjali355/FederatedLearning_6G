"""
Trust Score Prediction for Federated Learning
Predicts future client trust scores based on historical patterns
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class TrustPredictor:
    """
    Predict future trust scores using machine learning
    """
    
    def __init__(self, 
                 model_type: str = 'random_forest',
                 lookback_window: int = 5,
                 prediction_horizon: int = 3):
        """
        Args:
            model_type: 'random_forest', 'gradient_boost', 'ridge'
            lookback_window: Number of past rounds to consider
            prediction_horizon: Number of future rounds to predict
        """
        self.model_type = model_type
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        
        # Initialize model
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'gradient_boost':
            self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif model_type == 'ridge':
            self.model = Ridge(alpha=1.0)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_features(self, trust_history: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create features from trust score history
        
        Args:
            trust_history: Array of shape (num_clients, num_rounds)
        
        Returns:
            X (features), y (targets)
        """
        num_clients, num_rounds = trust_history.shape
        
        if num_rounds < self.lookback_window + self.prediction_horizon:
            raise ValueError("Not enough historical data")
        
        X_list = []
        y_list = []
        
        for client_idx in range(num_clients):
            client_scores = trust_history[client_idx]
            
            # Create sliding windows
            for t in range(num_rounds - self.lookback_window - self.prediction_horizon + 1):
                # Features: past lookback_window scores
                features = client_scores[t:t + self.lookback_window]
                
                # Add derived features
                derived = self._compute_derived_features(features)
                features = np.concatenate([features, derived])
                
                # Target: next prediction_horizon scores
                target = client_scores[t + self.lookback_window:
                                      t + self.lookback_window + self.prediction_horizon]
                
                X_list.append(features)
                y_list.append(target)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        return X, y
    
    def _compute_derived_features(self, scores: np.ndarray) -> np.ndarray:
        """
        Compute derived features from raw scores
        
        Features:
        - Mean, std, min, max
        - Trend (linear regression slope)
        - Volatility (rolling std)
        - Acceleration (second derivative)
        """
        features = []
        
        # Statistical features
        features.append(np.mean(scores))
        features.append(np.std(scores))
        features.append(np.min(scores))
        features.append(np.max(scores))
        
        # Trend
        if len(scores) > 1:
            x = np.arange(len(scores))
            trend = np.polyfit(x, scores, 1)[0]
            features.append(trend)
        else:
            features.append(0.0)
        
        # Volatility
        if len(scores) > 2:
            volatility = np.std(np.diff(scores))
            features.append(volatility)
        else:
            features.append(0.0)
        
        # Acceleration (change in trend)
        if len(scores) > 2:
            diffs = np.diff(scores)
            acceleration = np.mean(np.diff(diffs)) if len(diffs) > 1 else 0.0
            features.append(acceleration)
        else:
            features.append(0.0)
        
        return np.array(features)
    
    def train(self, trust_history: np.ndarray):
        """
        Train prediction model on historical data
        
        Args:
            trust_history: Array of shape (num_clients, num_rounds)
        """
        X, y = self.prepare_features(trust_history)
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # For multi-output prediction, train on mean of future scores
        y_mean = np.mean(y, axis=1)
        
        # Train model
        self.model.fit(X_scaled, y_mean)
        self.is_trained = True
    
    def predict(self, recent_scores: np.ndarray) -> float:
        """
        Predict future trust score
        
        Args:
            recent_scores: Recent scores (length = lookback_window)
        
        Returns:
            Predicted future trust score
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        if len(recent_scores) != self.lookback_window:
            raise ValueError(f"Need exactly {self.lookback_window} recent scores")
        
        # Compute features
        derived = self._compute_derived_features(recent_scores)
        features = np.concatenate([recent_scores, derived]).reshape(1, -1)
        
        # Standardize
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        
        # Clip to valid range
        return np.clip(prediction, 0.0, 1.0)
    
    def predict_trajectory(self, recent_scores: np.ndarray, num_steps: int = 5) -> np.ndarray:
        """
        Predict trust score trajectory for multiple future rounds
        
        Args:
            recent_scores: Recent scores (length = lookback_window)
            num_steps: Number of future steps to predict
        
        Returns:
            Array of predicted scores
        """
        predictions = []
        current_scores = recent_scores.copy()
        
        for _ in range(num_steps):
            # Predict next score
            next_score = self.predict(current_scores)
            predictions.append(next_score)
            
            # Shift window
            current_scores = np.roll(current_scores, -1)
            current_scores[-1] = next_score
        
        return np.array(predictions)


class EarlyWarningSystem:
    """
    Detect clients with degrading trust scores
    """
    
    def __init__(self, 
                 warning_threshold: float = 0.5,
                 decline_threshold: float = -0.1):
        """
        Args:
            warning_threshold: Trust score below which to warn
            decline_threshold: Rate of decline to trigger warning
        """
        self.warning_threshold = warning_threshold
        self.decline_threshold = decline_threshold
    
    def detect_degradation(self, trust_scores: np.ndarray) -> Dict[str, any]:
        """
        Analyze trust score trajectory for warning signs
        
        Args:
            trust_scores: Recent trust scores
        
        Returns:
            Warning information
        """
        if len(trust_scores) < 2:
            return {'warning': False, 'reason': 'insufficient_data'}
        
        current_score = trust_scores[-1]
        
        # Check absolute threshold
        if current_score < self.warning_threshold:
            return {
                'warning': True,
                'reason': 'low_trust',
                'severity': 'high' if current_score < 0.3 else 'medium',
                'current_score': current_score
            }
        
        # Check decline rate
        if len(trust_scores) >= 3:
            recent_trend = np.polyfit(np.arange(len(trust_scores)), trust_scores, 1)[0]
            
            if recent_trend < self.decline_threshold:
                return {
                    'warning': True,
                    'reason': 'declining_trust',
                    'severity': 'medium',
                    'decline_rate': recent_trend,
                    'current_score': current_score
                }
        
        # Check volatility
        if len(trust_scores) >= 5:
            volatility = np.std(trust_scores[-5:])
            
            if volatility > 0.2:
                return {
                    'warning': True,
                    'reason': 'high_volatility',
                    'severity': 'low',
                    'volatility': volatility,
                    'current_score': current_score
                }
        
        return {'warning': False, 'current_score': current_score}
    
    def prioritize_clients(self, all_trust_scores: Dict[int, np.ndarray]) -> List[Tuple[int, Dict]]:
        """
        Rank clients by warning priority
        
        Args:
            all_trust_scores: Dict mapping client_id to trust score history
        
        Returns:
            List of (client_id, warning_info) sorted by severity
        """
        warnings = []
        
        for client_id, scores in all_trust_scores.items():
            warning_info = self.detect_degradation(scores)
            if warning_info['warning']:
                warnings.append((client_id, warning_info))
        
        # Sort by severity
        severity_order = {'high': 0, 'medium': 1, 'low': 2}
        warnings.sort(key=lambda x: severity_order.get(x[1].get('severity', 'low'), 3))
        
        return warnings


class TrustAnomalyDetection:
    """
    Detect anomalous trust score patterns
    """
    
    def __init__(self, contamination: float = 0.1):
        """
        Args:
            contamination: Expected proportion of anomalies
        """
        self.contamination = contamination
        self.baseline_mean = None
        self.baseline_std = None
    
    def fit_baseline(self, trust_history: np.ndarray):
        """
        Establish baseline from historical data
        
        Args:
            trust_history: Array of shape (num_clients, num_rounds)
        """
        self.baseline_mean = np.mean(trust_history)
        self.baseline_std = np.std(trust_history)
    
    def detect_anomalies(self, current_scores: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """
        Detect anomalous trust scores using z-score
        
        Args:
            current_scores: Current round's trust scores
            threshold: Z-score threshold
        
        Returns:
            Boolean array indicating anomalies
        """
        if self.baseline_mean is None:
            # Use current data for baseline
            self.baseline_mean = np.mean(current_scores)
            self.baseline_std = np.std(current_scores)
        
        z_scores = np.abs((current_scores - self.baseline_mean) / (self.baseline_std + 1e-8))
        anomalies = z_scores > threshold
        
        return anomalies
    
    def detect_sudden_changes(self, previous_scores: np.ndarray, current_scores: np.ndarray, 
                             threshold: float = 0.3) -> np.ndarray:
        """
        Detect sudden changes in trust scores
        
        Args:
            previous_scores: Previous round scores
            current_scores: Current round scores
            threshold: Change threshold
        
        Returns:
            Boolean array indicating sudden changes
        """
        changes = np.abs(current_scores - previous_scores)
        sudden_changes = changes > threshold
        
        return sudden_changes


def generate_trust_report(predictor: TrustPredictor, 
                         early_warning: EarlyWarningSystem,
                         trust_history: Dict[int, np.ndarray]) -> Dict:
    """
    Generate comprehensive trust analysis report
    
    Args:
        predictor: Trained trust predictor
        early_warning: Early warning system
        trust_history: Historical trust scores per client
    
    Returns:
        Comprehensive report
    """
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'num_clients': len(trust_history),
        'warnings': [],
        'predictions': {},
        'statistics': {}
    }
    
    # Get warnings
    warnings = early_warning.prioritize_clients(trust_history)
    report['warnings'] = [
        {'client_id': cid, **info} for cid, info in warnings
    ]
    
    # Generate predictions
    for client_id, scores in trust_history.items():
        if len(scores) >= predictor.lookback_window:
            try:
                recent = scores[-predictor.lookback_window:]
                predicted = predictor.predict_trajectory(recent, num_steps=3)
                report['predictions'][client_id] = predicted.tolist()
            except:
                pass
    
    # Overall statistics
    all_scores = np.concatenate([scores for scores in trust_history.values()])
    report['statistics'] = {
        'mean_trust': float(np.mean(all_scores)),
        'std_trust': float(np.std(all_scores)),
        'min_trust': float(np.min(all_scores)),
        'max_trust': float(np.max(all_scores)),
        'num_warnings': len(warnings)
    }
    
    return report
