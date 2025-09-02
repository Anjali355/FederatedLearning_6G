import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# Option 1: Regularized MLP (Recommended for your use case)
class RegularizedMLP(nn.Module):
    def __init__(self, d_in: int, dropout_rate=0.3, use_batch_norm=True):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        
        self.fc1 = nn.Linear(d_in, 128)
        self.bn1 = nn.BatchNorm1d(128) if use_batch_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64) if use_batch_norm else nn.Identity()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32) if use_batch_norm else nn.Identity()
        self.dropout3 = nn.Dropout(dropout_rate / 2)
        
        self.fc4 = nn.Linear(32, 2)
        
        # Initialize weights properly
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        return self.fc4(x)

# Option 2: Robust MLP with Gradient Clipping Built-in
class MLP(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        # Wider network for better representation
        self.net = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
        # Apply weight initialization
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        return self.net(x)
    
    def get_gradient_norm(self):
        """Calculate gradient norm for anomaly detection"""
        total_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** (1. / 2)

# Option 3: Ensemble-like MLP for Better Detection
class EnsembleMLP(nn.Module):
    def __init__(self, d_in: int, num_heads=3):
        super().__init__()
        self.num_heads = num_heads
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(d_in, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Multiple prediction heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2)
            ) for _ in range(num_heads)
        ])
        
    def forward(self, x):
        features = self.feature_extractor(x)
        outputs = [head(features) for head in self.heads]
        # Average ensemble predictions
        return torch.mean(torch.stack(outputs), dim=0)
    
    def get_head_disagreement(self, x):
        """Measure disagreement between heads - useful for uncertainty estimation"""
        features = self.feature_extractor(x)
        outputs = [F.softmax(head(features), dim=1) for head in self.heads]
        stacked = torch.stack(outputs)
        variance = torch.var(stacked, dim=0).mean()
        return variance.item()

# Enhanced Training Function with Better Metrics
def train_with_enhanced_metrics(model, train_loader, val_loader, optimizer, criterion, device):
    """Training function that provides better metrics for malicious detection"""
    
    model.train()
    epoch_loss = 0.0
    total_samples = 0
    gradient_norms = []
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        
        # Calculate gradient norm before clipping
        grad_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        gradient_norms.append(grad_norm)
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        total_samples += len(data)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    correct = 0
    val_samples = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            val_samples += len(data)
    
    return {
        'train_loss': epoch_loss / len(train_loader),
        'val_loss': val_loss / len(val_loader),
        'val_accuracy': correct / val_samples,
        'avg_grad_norm': sum(gradient_norms) / len(gradient_norms),
        'max_grad_norm': max(gradient_norms),
        'grad_norm_variance': torch.tensor(gradient_norms).var().item()
    }

# Enhanced Trust Calculation Using Model-Specific Metrics
def calculate_model_based_trust(model, metrics, client_id):
    """Calculate trust score using model-specific features"""
    
    base_trust = 1.0
    
    # 1. Gradient-based detection
    avg_grad_norm = metrics.get('avg_grad_norm', 0)
    max_grad_norm = metrics.get('max_grad_norm', 0)
    grad_variance = metrics.get('grad_norm_variance', 0)
    
    # Penalize unusual gradient patterns
    if max_grad_norm > 5.0:  # Abnormally high gradients
        base_trust *= 0.3
    if grad_variance > 2.0:  # Highly variable gradients
        base_trust *= 0.7
    
    # 2. Performance inconsistency
    train_loss = metrics.get('train_loss', float('inf'))
    val_loss = metrics.get('val_loss', float('inf'))
    val_acc = metrics.get('val_accuracy', 0)
    
    # Loss divergence (overfitting indicator)
    if train_loss < 0.5 and val_loss > 1.5:
        base_trust *= 0.4
    
    # Poor generalization
    if val_acc < 0.5:
        base_trust *= 0.6
    
    # 3. Model-specific anomalies (if using ensemble)
    if hasattr(model, 'get_head_disagreement'):
        disagreement = model.get_head_disagreement(torch.randn(1, model.feature_extractor[0].in_features))
        if disagreement > 0.5:  # High uncertainty
            base_trust *= 0.8
    
    return max(0.01, min(1.0, base_trust))

def set_model_params(model: nn.Module, params: List) -> None:
    sd = model.state_dict()
    new_sd = {}
    for (k, v), arr in zip(sd.items(), params):
        new_sd[k] = torch.tensor(arr)
    model.load_state_dict(new_sd, strict=True)

def get_model_params(model: nn.Module) -> List:
    return [p.detach().cpu().numpy() for p in model.state_dict().values()]