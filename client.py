import torch
import flwr as fl
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn as nn

from model import MLP, get_model_params, set_model_params

@dataclass
class ClientConfig:
    cid: int
    X: np.ndarray
    y: np.ndarray
    device: torch.device
    dp_sigma: float
    clip_norm: float
    local_epochs: int
    batch_size: int
    lr: float
    malicious: bool

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cfg: ClientConfig, d_in: int):
        self.cfg = cfg
        X = torch.tensor(cfg.X, dtype=torch.float32)
        y = torch.tensor(cfg.y, dtype=torch.long)
        ds = TensorDataset(X, y)

        n = len(ds)
        n_val = max(int(0.2 * n), 1)  # Increased validation split to 20%
        n_tr = max(n - n_val, 1)
        self.train_ds, self.val_ds = random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(123+cfg.cid))

        self.model = MLP(d_in=d_in).to(cfg.device)
        self.last_grad_norm = 0.0

        print(f"[CLIENT {cfg.cid}] Train samples: {len(self.train_ds)}, Val samples: {len(self.val_ds)}")

    def get_parameters(self, config):
        return get_model_params(self.model)

    def fit(self, parameters, config):
        set_model_params(self.model, parameters)
        self.model.train()
        
        # Use a higher learning rate for better convergence
        effective_lr = max(self.cfg.lr, 0.01)  # Ensure minimum lr of 0.01
        opt = torch.optim.Adam(self.model.parameters(), lr=effective_lr)
        criterion = nn.CrossEntropyLoss()

        flip_prob = 0.3 if self.cfg.malicious else 0.0
        train_loader = DataLoader(self.train_ds, batch_size=self.cfg.batch_size, shuffle=True)

        total_train_loss = 0.0
        total_batches = 0
        
        for epoch in range(self.cfg.local_epochs):
            epoch_loss = 0.0
            epoch_batches = 0
            
            for xb, yb in train_loader:
                xb = xb.to(self.cfg.device)
                yb = yb.clone()
                
                # Apply label flipping for malicious clients
                if flip_prob > 0:
                    mask = torch.rand_like(yb.float()) < flip_prob
                    yb[mask] = 1 - yb[mask]
                yb = yb.to(self.cfg.device)

                opt.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()

                # Gradient clipping
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_norm).item()
                self.last_grad_norm = float(total_norm)

                # Differential privacy noise
                if self.cfg.dp_sigma > 0:
                    with torch.no_grad():
                        for p in self.model.parameters():
                            if p.grad is not None:
                                noise = torch.normal(
                                    mean=0.0,
                                    std=self.cfg.dp_sigma * self.cfg.clip_norm,
                                    size=p.grad.shape,
                                    device=p.grad.device,
                                )
                                p.grad.add_(noise)

                opt.step()
                
                epoch_loss += loss.item()
                epoch_batches += 1
            
            if epoch_batches > 0:
                total_train_loss += epoch_loss / epoch_batches
                total_batches += 1

        # Calculate average training loss
        avg_train_loss = total_train_loss / max(total_batches, 1)
        
        # Get validation metrics
        val_acc, val_loss = self._validate()
        
        # Calculate training accuracy for comparison
        train_acc = self._calculate_train_accuracy()
        
        print(f"[CLIENT {self.cfg.cid}] Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f}, Train loss: {avg_train_loss:.4f}, Grad norm: {self.last_grad_norm:.4f}")
        
        metrics = {
            "val_acc": float(val_acc), 
            "val_loss": float(val_loss), 
            "train_loss": float(avg_train_loss),
            "train_acc": float(train_acc),
            "grad_norm": float(self.last_grad_norm)
        }
        
        return get_model_params(self.model), len(self.train_ds), metrics

    def evaluate(self, parameters, config):
        set_model_params(self.model, parameters)
        self.model.eval()
        acc, loss = self._validate()
        return float(loss), len(self.val_ds), {"val_acc": float(acc)}

    def _validate(self) -> Tuple[float, float]:
        if len(self.val_ds) == 0:
            return 0.0, float('inf')
            
        loader = DataLoader(self.val_ds, batch_size=min(256, len(self.val_ds)), shuffle=False)
        correct, total = 0, 0
        crit = nn.CrossEntropyLoss(reduction='sum')
        loss_sum = 0.0
        
        self.model.eval()
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.cfg.device)
                yb = yb.to(self.cfg.device)
                logits = self.model(xb)
                loss = crit(logits, yb)
                loss_sum += loss.item()
                pred = logits.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
        
        acc = correct / max(total, 1)
        loss = loss_sum / max(total, 1)
        return acc, loss
    
    def _calculate_train_accuracy(self) -> float:
        """Calculate accuracy on training set for monitoring"""
        if len(self.train_ds) == 0:
            return 0.0
            
        loader = DataLoader(self.train_ds, batch_size=min(256, len(self.train_ds)), shuffle=False)
        correct, total = 0, 0
        
        self.model.eval()
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.cfg.device)
                yb = yb.to(self.cfg.device)
                logits = self.model(xb)
                pred = logits.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
        
        return correct / max(total, 1)