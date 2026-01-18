# Enhanced Strategy with Better Detection
import os, csv
import numpy as np
from scipy.spatial.distance import cosine, euclidean
from scipy import stats
from collections import defaultdict 
import torch
import flwr as fl
from sklearn.metrics import accuracy_score, log_loss

from model import MLP, set_model_params
from byzantine_defense import ByzantineDefense, detect_byzantine_clients
from differential_privacy import DifferentialPrivacy, AdaptivePrivacy
from trust_prediction import TrustPredictor, EarlyWarningSystem

class TrustFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, testset, device, outdir, 
                 aggregation_method='fedavg',
                 use_differential_privacy=False,
                 dp_epsilon=1.0,
                 dp_delta=1e-5,
                 use_trust_prediction=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.X_test, self.y_test = testset
        self.device = device
        self.outdir = outdir
        self.client_stats = {}
        self.trust_scores = {}
        self.parameter_history = defaultdict(list)
        self.global_model_history = []
        self.round_metrics = defaultdict(list)  # Track metrics over time
        self.detection_history = defaultdict(list)  # Track detection results
        self.blocked_clients = set()  # Track permanently blocked clients
        
        # Priority 1: Byzantine Defense
        self.aggregation_method = aggregation_method
        self.byzantine_defense = ByzantineDefense(num_byzantine=2)  # Assumes ~10-15% malicious
        
        # Priority 1: Differential Privacy
        self.use_differential_privacy = use_differential_privacy
        self.adaptive_privacy = AdaptivePrivacy(base_epsilon=dp_epsilon)
        self.dp_mechanisms = {}  # Per-client DP mechanisms
        
        # Priority 1: Trust Prediction
        self.use_trust_prediction = use_trust_prediction
        if use_trust_prediction:
            self.trust_predictor = TrustPredictor(lookback_window=5)
            self.early_warning = EarlyWarningSystem()
        else:
            self.trust_predictor = None
            self.early_warning = None
        
        os.makedirs(outdir, exist_ok=True)
        self.logfile = os.path.join(outdir, "enhanced_trust_log.csv")
        with open(self.logfile, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "round", "client_id", "test_acc", "test_loss", "trust_score", 
                "is_malicious_detected", "grad_norm", "param_divergence", 
                "cosine_similarity", "performance_variance", "label_flip_score",
                "predicted_trust", "dp_epsilon", "aggregation_method"
            ])

    def aggregate_fit(self, rnd, results, failures):
        print(f"\n=== ENHANCED DETECTION ROUND {rnd} ===")
        print(f"Participants: {len(results)}, Failures: {len(failures)}")
        
        if len(results) == 0:
            print("[ERROR] No clients participated")
            return None, {}
        
        # Extract client data
        client_data = []
        client_parameters = {}
        
        for i, (cid, fitres) in enumerate(results):
            client_id = str(i) if hasattr(cid, '__dict__') else str(cid)
            client_data.append((client_id, fitres))
            client_parameters[client_id] = fl.common.parameters_to_ndarrays(fitres.parameters)
            self.parameter_history[client_id].append(client_parameters[client_id])
        
        # ENHANCED DETECTION PIPELINE
        detection_results = {}
        for client_id, fitres in client_data:
            metrics = fitres.metrics
            params = client_parameters[client_id]
            
            # Multi-layered detection
            detection_score = self._comprehensive_malicious_detection(
                client_id, metrics, params, rnd
            )
            
            detection_results[client_id] = detection_score
            
            # Store metrics for temporal analysis
            self.round_metrics[client_id].append({
                'round': rnd,
                'grad_norm': metrics.get('grad_norm', 0),
                'val_acc': metrics.get('val_acc', 0),
                'train_acc': metrics.get('train_acc', 0),
                'train_loss': metrics.get('train_loss', 0),
                'detection_score': detection_score
            })
        
        # Calculate trust scores with enhanced detection
        for client_id in detection_results:
            detection_score = detection_results[client_id]
            
            # Convert detection score to trust score (inverse relationship)
            trust_score = 1.0 - detection_score
            
            # Apply temporal smoothing (less aggressive)
            if client_id in self.trust_scores:
                trust_score = 0.3 * self.trust_scores[client_id] + 0.7 * trust_score
            
            self.trust_scores[client_id] = max(0.01, min(1.0, trust_score))
            
            # Determine status with stricter thresholds
            is_malicious = detection_score > 0.3  # Lower threshold for better detection
            is_suspicious = detection_score > 0.15 #2
            
            self.client_stats[client_id] = {
                **fitres.metrics,
                "trust_score": trust_score,
                "detection_score": detection_score,
                "is_malicious": is_malicious,
                "is_suspicious": is_suspicious,
            }
            
            print(f"[DETECTION] Client {client_id}: detection={detection_score:.3f}, "
                  f"trust={trust_score:.3f}, status={'MALICIOUS' if is_malicious else 'SUSPICIOUS' if is_suspicious else 'NORMAL'}")
        
        # Enhanced weight calculation with stronger penalties
        weights_results = []
        total_examples = sum(fitres.num_examples for _, fitres in client_data)
        
        for client_id, fitres in client_data:
            base_weight = fitres.num_examples / total_examples
            
            # Aggressive trust-based weighting
            trust_score = self.trust_scores[client_id]
            if trust_score < 0.7:  # Likely malicious
                adjusted_weight = base_weight * 0.01  # Nearly exclude
            elif trust_score < 0.85:  # Suspicious
                adjusted_weight = base_weight * 0.1   # Heavily penalize
            elif trust_score < 0.9:  # Questionable
                adjusted_weight = base_weight * 0.5   # Moderate penalty
            else:  # Trusted
                adjusted_weight = base_weight * trust_score
            
            weights_results.append((fitres.parameters, adjusted_weight))
            print(f"[WEIGHTS] Client {client_id}: {base_weight:.4f} -> {adjusted_weight:.4f} "
                  f"(reduction: {(1-adjusted_weight/base_weight)*100:.1f}%)")
        
        # Aggregation with Byzantine Defense
        param_arrays = [fl.common.parameters_to_ndarrays(params) for params, _ in weights_results]
        weights = [weight for _, weight in weights_results]
        
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            print("[ERROR] All clients excluded due to low trust")
            return None, {}
        
        # Apply Byzantine-robust aggregation
        print(f"\n[AGGREGATION] Using method: {self.aggregation_method}")
        
        # Flatten each client's parameters for Byzantine detection
        shapes = [arr.shape for arr in param_arrays[0]]
        flattened_params = []
        for client_params in param_arrays:
            flat = np.concatenate([arr.flatten() for arr in client_params])
            flattened_params.append(flat)
        
        if self.aggregation_method in ['krum', 'multi-krum', 'trimmed-mean', 'median']:
            # Apply Byzantine-robust method
            if self.aggregation_method == 'krum':
                flat_aggregated = self.byzantine_defense.krum(flattened_params)
                print("[AGGREGATION] Applied Krum - selected most representative update")
            elif self.aggregation_method == 'multi-krum':
                flat_aggregated = self.byzantine_defense.multi_krum(flattened_params, m=5)
                print("[AGGREGATION] Applied Multi-Krum - averaged top 5 updates")
            elif self.aggregation_method == 'trimmed-mean':
                flat_aggregated = self.byzantine_defense.trimmed_mean(flattened_params, beta=2)
                print("[AGGREGATION] Applied Trimmed Mean - removed extremes")
            elif self.aggregation_method == 'median':
                flat_aggregated = self.byzantine_defense.coordinate_wise_median(flattened_params)
                print("[AGGREGATION] Applied Coordinate-wise Median")
            
            # Reconstruct layer structure
            aggregated_arrays = []
            start_idx = 0
            for shape in shapes:
                size = np.prod(shape)
                layer_flat = flat_aggregated[start_idx:start_idx + size]
                aggregated_arrays.append(layer_flat.reshape(shape))
                start_idx += size
        else:  # 'fedavg' or default
            # Traditional weighted average
            aggregated_arrays = []
            for layer_idx in range(len(param_arrays[0])):
                layer_arrays = [client_params[layer_idx] for client_params in param_arrays]
                weighted_sum = np.zeros_like(layer_arrays[0])
                for arr, weight in zip(layer_arrays, weights):
                    weighted_sum += arr * weight
                aggregated_arrays.append(weighted_sum)
            print("[AGGREGATION] Applied standard FedAvg")
        
        # Detect Byzantine clients
        byzantine_detected = detect_byzantine_clients(flattened_params)
        if len(byzantine_detected) > 0:
            print(f"[BYZANTINE DETECTION] Potential Byzantine clients: {byzantine_detected}")
            for idx in byzantine_detected:
                cid = list(client_data)[idx][0]
                if cid in self.trust_scores:
                    self.trust_scores[cid] *= 0.5  # Penalize
        
        # Calculate robustness score
        robustness = self.byzantine_defense.get_robustness_score(flattened_params)
        print(f"[ROBUSTNESS] Update dispersion score: {robustness:.4f}")
        
        aggregated_parameters = fl.common.ndarrays_to_parameters(aggregated_arrays)
        
        # Evaluation and logging
        test_acc, test_loss = self.evaluate_global(aggregated_arrays)
        malicious_detected = [cid for cid, st in self.client_stats.items() if st["is_malicious"]]
        suspicious_detected = [cid for cid, st in self.client_stats.items() if st["is_suspicious"]]
        
        print(f"[ROUND {rnd}] Test Acc: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
        print(f"[DETECTION] Malicious: {malicious_detected}, Suspicious: {suspicious_detected}")
        
        # Trust Prediction (Priority 1)
        trust_predictions = {}
        if self.use_trust_prediction and rnd >= 5:
            # Train predictor on historical data
            try:
                trust_history_matrix = self._build_trust_history_matrix()
                if trust_history_matrix.shape[1] >= 5:
                    self.trust_predictor.train(trust_history_matrix)
                    
                    # Predict future trust for each client
                    for client_id in self.trust_scores:
                        recent_scores = [m['detection_score'] for m in self.round_metrics[client_id][-5:]]
                        if len(recent_scores) >= 5:
                            recent_array = np.array(recent_scores)
                            predicted_trust = self.trust_predictor.predict(recent_array)
                            trust_predictions[client_id] = predicted_trust
                    
                    # Early warning system
                    warnings = self.early_warning.prioritize_clients(
                        {cid: np.array([1.0 - m['detection_score'] for m in self.round_metrics[cid]]) 
                         for cid in self.trust_scores}
                    )
                    if warnings:
                        print(f"\n[EARLY WARNING] {len(warnings)} clients with degrading trust:")
                        for cid, warning_info in warnings[:3]:  # Show top 3
                            print(f"  Client {cid}: {warning_info['reason']} "
                                  f"(severity: {warning_info['severity']})")
            except Exception as e:
                print(f"[WARNING] Trust prediction failed: {e}")
        
        # Enhanced logging with Priority 1 features
        for client_id in detection_results:
            with open(self.logfile, "a", newline="") as f:
                writer = csv.writer(f)
                stats = self.client_stats[client_id]
                
                # Get DP epsilon if used
                dp_eps = 0.0
                if self.use_differential_privacy and client_id in self.dp_mechanisms:
                    dp_eps = self.dp_mechanisms[client_id].epsilon
                
                # Get predicted trust
                pred_trust = trust_predictions.get(client_id, -1.0)
                
                writer.writerow([
                    rnd, client_id, test_acc, test_loss, stats["trust_score"],
                    stats["is_malicious"], stats.get("grad_norm", 0),
                    detection_results[client_id], "computed", "computed", "computed",
                    pred_trust, dp_eps, self.aggregation_method
                ])
        
        self.trust_weights = self.trust_scores.copy()
        return aggregated_parameters, {}
    
    def _build_trust_history_matrix(self):
        """Build matrix of trust scores over time for prediction"""
        client_ids = sorted(self.trust_scores.keys())
        max_rounds = max(len(self.round_metrics[cid]) for cid in client_ids)
        
        matrix = np.zeros((len(client_ids), max_rounds))
        for i, cid in enumerate(client_ids):
            for j, metrics in enumerate(self.round_metrics[cid]):
                matrix[i, j] = 1.0 - metrics['detection_score']  # Convert to trust
        
        return matrix

    def _comprehensive_malicious_detection(self, client_id, metrics, params, round_num):
        """Multi-factor malicious client detection"""
        
        detection_scores = []
        
        # 1. GRADIENT ANOMALY DETECTION (Enhanced)
        grad_norm = metrics.get('grad_norm', 0.0)
        if grad_norm > 2.0:  # Stricter threshold
            grad_score = min(1.0, (grad_norm - 2.0) / 3.0)
        elif grad_norm < 0.01:  # Suspiciously low gradients
            grad_score = 0.3
        else:
            grad_score = 0.0
        detection_scores.append(('gradient_anomaly', grad_score))
        
        # 2. PERFORMANCE INCONSISTENCY (Enhanced)
        train_acc = metrics.get('train_acc', 0.0)
        val_acc = metrics.get('val_acc', 0.0)
        train_loss = metrics.get('train_loss', float('inf'))
        
        # Classic overfitting pattern (malicious clients often overfit on poisoned data)
        if train_acc > 0.7 and val_acc < 0.4:
            perf_score = 0.8
        elif abs(train_acc - val_acc) > 0.3:  # Large gap
            perf_score = 0.6
        elif val_acc < 0.3:  # Very poor validation
            perf_score = 0.7
        else:
            perf_score = 0.0
        detection_scores.append(('performance_inconsistency', perf_score))
        
        # 3. PARAMETER DIVERGENCE (Enhanced)
        if round_num > 1 and len(self.parameter_history[client_id]) >= 2:
            divergence = self._calculate_enhanced_divergence(client_id, params)
            if divergence > 1.5:  # Stricter threshold
                div_score = min(1.0, (divergence - 1.5) / 2.0)
            else:
                div_score = 0.0
            detection_scores.append(('parameter_divergence', div_score))
        
        # 4. STATISTICAL OUTLIER DETECTION (New)
        if round_num > 2:
            outlier_score = self._statistical_outlier_detection(client_id, metrics)
            detection_scores.append(('statistical_outlier', outlier_score))
        
        # 5. COSINE SIMILARITY ANALYSIS (Enhanced)
        if len(self.parameter_history) > 2:
            similarity_score = self._enhanced_similarity_analysis(client_id, params, round_num)
            detection_scores.append(('similarity_anomaly', similarity_score))
        
        # 6. TEMPORAL PATTERN ANALYSIS (New)
        if round_num > 3:
            temporal_score = self._temporal_pattern_analysis(client_id)
            detection_scores.append(('temporal_anomaly', temporal_score))
        
        # Combine detection scores with weights
        weights = {
            'gradient_anomaly': 0.25,
            'performance_inconsistency': 0.25,
            'parameter_divergence': 0.20,
            'statistical_outlier': 0.15,
            'similarity_anomaly': 0.10,
            'temporal_anomaly': 0.05
        }
        
        final_score = sum(score * weights.get(method, 0.1) 
                         for method, score in detection_scores)
        
        # Debug output
        debug_info = {method: f"{score:.3f}" for method, score in detection_scores}
        print(f"[DETAILED-DETECTION] Client {client_id}: {debug_info} -> Final: {final_score:.3f}")
        
        return min(1.0, final_score)

    def _calculate_enhanced_divergence(self, client_id, current_params):
        """Enhanced parameter divergence calculation"""
        if len(self.parameter_history[client_id]) < 2:
            return 0.0
        
        previous_params = self.parameter_history[client_id][-2]
        
        # Calculate multiple divergence metrics
        l2_changes = []
        cosine_distances = []
        
        for curr, prev in zip(current_params, previous_params):
            # L2 norm of change
            change_norm = np.linalg.norm(curr - prev)
            param_norm = np.linalg.norm(prev)
            relative_change = change_norm / (param_norm + 1e-8)
            l2_changes.append(relative_change)
            
            # Cosine distance
            curr_flat = curr.flatten()
            prev_flat = prev.flatten()
            if len(curr_flat) > 1:  # Avoid issues with bias terms
                cos_dist = cosine(curr_flat, prev_flat)
                cosine_distances.append(cos_dist)
        
        # Combine metrics
        avg_l2_change = np.mean(l2_changes)
        avg_cosine_dist = np.mean(cosine_distances) if cosine_distances else 0
        
        # Weighted combination
        divergence = 0.7 * avg_l2_change + 0.3 * avg_cosine_dist
        return divergence

    def _statistical_outlier_detection(self, client_id, metrics):
        """Detect statistical outliers using z-scores"""
        if len(self.round_metrics[client_id]) < 3:
            return 0.0
        
        # Get historical metrics for this client
        history = self.round_metrics[client_id]
        current_metrics = {
            'grad_norm': metrics.get('grad_norm', 0),
            'val_acc': metrics.get('val_acc', 0),
            'train_loss': metrics.get('train_loss', 0)
        }
        
        outlier_scores = []
        
        for metric_name, current_val in current_metrics.items():
            # Get historical values
            historical_vals = [h.get(metric_name, 0) for h in history]
            
            if len(historical_vals) >= 3:
                mean_val = np.mean(historical_vals)
                std_val = np.std(historical_vals)
                
                if std_val > 0:
                    z_score = abs(current_val - mean_val) / std_val
                    # Convert z-score to outlier probability
                    outlier_prob = min(1.0, max(0.0, (z_score - 2.0) / 2.0))
                    outlier_scores.append(outlier_prob)
        
        return np.mean(outlier_scores) if outlier_scores else 0.0

    def _enhanced_similarity_analysis(self, client_id, client_params, round_num):
        """Enhanced similarity analysis with robust statistics"""
        other_params = []
        
        for other_id, params_history in self.parameter_history.items():
            if other_id != client_id and len(params_history) >= round_num:
                other_params.append(params_history[round_num - 1])
        
        if len(other_params) < 2:
            return 0.0
        
        # Flatten current client parameters
        client_flat = np.concatenate([p.flatten() for p in client_params])
        
        # Calculate similarities with all other clients
        similarities = []
        for other_param_set in other_params:
            other_flat = np.concatenate([p.flatten() for p in other_param_set])
            
            # Use multiple similarity metrics
            cosine_sim = 1 - cosine(client_flat, other_flat)
            
            # Euclidean distance (normalized)
            euclidean_dist = euclidean(client_flat, other_flat)
            max_norm = max(np.linalg.norm(client_flat), np.linalg.norm(other_flat))
            normalized_euclidean = euclidean_dist / (max_norm + 1e-8)
            
            similarities.append({
                'cosine': cosine_sim,
                'euclidean': normalized_euclidean
            })
        
        # Statistical analysis of similarities
        cosine_sims = [s['cosine'] for s in similarities]
        euclidean_dists = [s['euclidean'] for s in similarities]
        
        # Check if client is an outlier
        cosine_mean = np.mean(cosine_sims)
        cosine_std = np.std(cosine_sims)
        
        # Low similarity (outlier) indicates potential malicious behavior
        if cosine_mean < 0.5:  # Very dissimilar to others
            similarity_score = 0.8
        elif cosine_mean < 0.7:  # Somewhat dissimilar
            similarity_score = 0.4
        else:
            similarity_score = 0.0
        
        return similarity_score

    def _temporal_pattern_analysis(self, client_id):
        """Analyze temporal patterns in client behavior"""
        if len(self.round_metrics[client_id]) < 4:
            return 0.0
        
        history = self.round_metrics[client_id]
        
        # Check for suspicious patterns
        val_accs = [h['val_acc'] for h in history]
        grad_norms = [h['grad_norm'] for h in history]
        
        pattern_scores = []
        
        # Pattern 1: Consistently decreasing validation accuracy
        if len(val_accs) >= 3:
            trend = np.polyfit(range(len(val_accs)), val_accs, 1)[0]
            if trend < -0.1:  # Decreasing trend
                pattern_scores.append(0.6)
        
        # Pattern 2: Irregular gradient norms
        if len(grad_norms) >= 4:
            grad_variance = np.var(grad_norms)
            if grad_variance > 1.0:  # High variance in gradients
                pattern_scores.append(0.5)
        
        # Pattern 3: Sudden performance changes
        recent_metrics = history[-3:]
        for i in range(1, len(recent_metrics)):
            curr_acc = recent_metrics[i]['val_acc']
            prev_acc = recent_metrics[i-1]['val_acc']
            if abs(curr_acc - prev_acc) > 0.4:  # Sudden change
                pattern_scores.append(0.7)
        
        return np.mean(pattern_scores) if pattern_scores else 0.0

    def evaluate_global(self, params):
        """Evaluate global model on test set"""
        model = MLP(d_in=self.X_test.shape[1]).to(self.device)
        set_model_params(model, params)
        model.eval()

        X = torch.tensor(self.X_test, dtype=torch.float32).to(self.device)
        y = torch.tensor(self.y_test, dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = model(X)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

        acc = accuracy_score(y.cpu().numpy(), preds)
        loss = log_loss(y.cpu().numpy(), probs, labels=[0, 1])
        return acc, loss

# ENHANCED MALICIOUS CLIENT IMPLEMENTATION
# Add this to your client.py to make malicious behavior more detectable:

def enhanced_malicious_behavior(self, model, data_loader, round_num):
    """More sophisticated malicious behavior for better testing"""
    
    # Strategy 1: Label Flipping with Gradual Increase
    flip_rate = min(0.5, 0.1 + 0.05 * round_num)  # Increase over time
    
    # Strategy 2: Model Poisoning - inject bad gradients
    for param in model.parameters():
        if param.grad is not None:
            # Add adversarial noise to gradients
            noise_scale = 0.1 + 0.05 * round_num
            adversarial_noise = torch.randn_like(param.grad) * noise_scale
            param.grad.data += adversarial_noise
    
    # Strategy 3: Fake metrics to avoid detection
    fake_train_acc = 0.7 + 0.1 * np.random.random()  # Fake good performance
    fake_val_acc = 0.3 + 0.1 * np.random.random()    # But poor validation
    
    return {
        'train_acc': fake_train_acc,
        'val_acc': fake_val_acc,
        'train_loss': 0.4 + 0.2 * np.random.random(),
        'grad_norm': 0.5 + np.random.random() * 2.0,  # Variable grad norms
        'attack_type': f'enhanced_malicious_round_{round_num}'
    }