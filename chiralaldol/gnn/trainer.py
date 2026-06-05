"""Unified GNN training loop for AldolRxnMaster.

Supports all 4 GNN architectures × 3 fusion modes with:
  - Early stopping
  - Label smoothing
  - Class-balanced loss
  - Cosine LR schedule
  - Time-series CV evaluation
"""

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

logger = logging.getLogger(__name__)


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing for classification."""

    def __init__(self, smoothing: float = 0.1, weight: torch.Tensor | None = None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, pred, target):
        n_class = pred.size(-1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class

        log_prob = F.log_softmax(pred, dim=-1)
        if self.weight is not None:
            log_prob = log_prob * self.weight.unsqueeze(0)

        loss = -(one_hot * log_prob).sum(dim=-1).mean()
        return loss


def compute_class_weights(labels: np.ndarray, num_classes: int = 4) -> torch.Tensor:
    """Compute inverse-frequency class weights."""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.maximum(counts, 1.0)  # avoid division by zero
    weights = len(labels) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float)


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_samples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, batch.y.squeeze())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        n_samples += batch.num_graphs

    return total_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model on a dataset."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(batch.y.squeeze().cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    bal_acc = balanced_accuracy_score(y_true, y_pred)
    return bal_acc, y_true, y_pred, y_prob


def train_and_evaluate(
    model: nn.Module,
    train_graphs: list[Data],
    val_graphs: list[Data],
    test_graphs: list[Data],
    config: dict,
    device: torch.device,
) -> dict:
    """Full training + evaluation pipeline.

    Args:
        model: GNN model
        train_graphs: list of PyG Data objects for training
        val_graphs: list for validation
        test_graphs: list for testing
        config: dict with lr, weight_decay, epochs, patience, batch_size, label_smoothing
        device: torch device

    Returns:
        dict with bal_acc, mcc, y_true, y_pred, y_prob, train_history
    """
    # Filter None graphs
    train_graphs = [g for g in train_graphs if g is not None]
    val_graphs = [g for g in val_graphs if g is not None]
    test_graphs = [g for g in test_graphs if g is not None]

    if not train_graphs or not val_graphs or not test_graphs:
        logger.warning("Empty graph sets — skipping")
        return {"bal_acc": 0.0, "error": "empty_graphs"}

    # Config
    lr = config.get("lr", 1e-3)
    weight_decay = config.get("weight_decay", 1e-4)
    epochs = config.get("epochs", 100)
    patience = config.get("patience", 20)
    batch_size = config.get("batch_size", 32)
    label_smoothing = config.get("label_smoothing", 0.1)

    # Data loaders
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True,
                               follow_batch=["x_r", "x_p"] if hasattr(train_graphs[0], "x_r") else [])
    val_loader = DataLoader(val_graphs, batch_size=batch_size,
                             follow_batch=["x_r", "x_p"] if hasattr(val_graphs[0], "x_r") else [])
    test_loader = DataLoader(test_graphs, batch_size=batch_size,
                              follow_batch=["x_r", "x_p"] if hasattr(test_graphs[0], "x_r") else [])

    # Class weights
    train_labels = np.array([g.y.item() for g in train_graphs])
    class_weights = compute_class_weights(train_labels).to(device)

    # Loss
    criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing, weight=class_weights)

    # Optimizer + scheduler
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    # Training loop with early stopping
    best_val_acc = 0
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_acc, _, _, _ = evaluate(model, val_loader, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"  Epoch {epoch:3d}: loss={train_loss:.4f}, "
                        f"val_acc={val_acc:.4f} (best={best_val_acc:.4f})")

        if patience_counter >= patience:
            logger.info(f"  Early stopping at epoch {epoch} (patience={patience})")
            break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation on test
    test_acc, y_true, y_pred, y_prob = evaluate(model, test_loader, device)

    return {
        "bal_acc": test_acc,
        "best_val_acc": best_val_acc,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "history": history,
        "epochs_trained": len(history["train_loss"]),
    }
