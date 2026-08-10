import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

# Cap threads at 8
torch.set_num_threads(8)
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))
from metric import CLASSES, balanced_accuracy, labels_from_proba, tune_decision_weights  # noqa

# Column definitions
NUM = ["sleep_duration", "heart_rate", "bmi", "calorie_expenditure",
       "step_count", "exercise_duration", "water_intake"]
CAT = ["diet_type", "stress_level", "sleep_quality", "physical_activity_level",
       "smoking_alcohol", "gender"]

class TabularMLP(nn.Module):
    def __init__(self, cat_dims, emb_dims, num_features, layers=[256, 128, 64], dropout=0.15):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_cat, emb_dim) for num_cat, emb_dim in zip(cat_dims, emb_dims)
        ])
        total_emb_dim = sum(emb_dims)
        input_dim = total_emb_dim + num_features
        
        mlp_layers = []
        in_dim = input_dim
        for out_dim in layers:
            mlp_layers.append(nn.Linear(in_dim, out_dim))
            mlp_layers.append(nn.BatchNorm1d(out_dim))
            mlp_layers.append(nn.SiLU())
            mlp_layers.append(nn.Dropout(dropout))
            in_dim = out_dim
            
        self.mlp = nn.Sequential(*mlp_layers)
        self.output = nn.Linear(in_dim, 3)

    def forward(self, x_cat, x_cont):
        emb_outs = []
        for i, emb in enumerate(self.embeddings):
            emb_outs.append(emb(x_cat[:, i]))
        x_emb = torch.cat(emb_outs, dim=1)
        x = torch.cat([x_emb, x_cont], dim=1)
        x = self.mlp(x)
        logits = self.output(x)
        return logits

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # tensor of shape (C,) representing weights per class
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def get_class_weights(y, device):
    _, counts = np.unique(y, return_counts=True)
    total = sum(counts)
    weights = total / (len(counts) * counts)
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32, device=device)

def get_loss_fn(loss_type, y_train, device, gamma=2.0):
    if loss_type == "ce":
        return nn.CrossEntropyLoss()
    elif loss_type == "weighted_ce":
        w = get_class_weights(y_train, device)
        print(f"  Class weights (normalized): {w.cpu().numpy().round(3)}")
        return nn.CrossEntropyLoss(weight=w)
    elif loss_type == "focal":
        return FocalLoss(gamma=gamma)
    elif loss_type == "weighted_focal":
        w = get_class_weights(y_train, device)
        print(f"  Class weights (normalized) for Focal: {w.cpu().numpy().round(3)}")
        return FocalLoss(alpha=w, gamma=gamma)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

def fast_tune_decision_weights(proba, y_true_str, n_iter=6000, seed=42):
    rng = np.random.default_rng(seed)
    class_to_idx = {"at-risk": 0, "fit": 1, "unhealthy": 2}
    y_true = np.array([class_to_idx[y] for y in y_true_str], dtype=np.int32)
    P = np.asarray(proba, dtype=float)
    
    class_masks = [y_true == c for c in range(3)]
    class_counts = [np.sum(mask) for mask in class_masks]
    
    def get_score(w):
        y_pred = (P * w[None, :]).argmax(axis=1)
        recalls = []
        for c in range(3):
            tp = np.sum((y_pred == c) & class_masks[c])
            recalls.append(tp / class_counts[c] if class_counts[c] > 0 else 0.0)
        return np.mean(recalls)
        
    best_w = np.ones(3)
    best_s = get_score(best_w)
    
    for _ in range(n_iter):
        w = np.exp(rng.normal(0.0, 0.7, size=3))
        s = get_score(w)
        if s > best_s:
            best_s, best_w = s, w
            
    step = 0.5
    for _ in range(300):
        improved = False
        for j in range(3):
            for d in (step, -step):
                w = best_w.copy()
                w[j] *= np.exp(d)
                s = get_score(w)
                if s > best_s:
                    best_s, best_w, improved = s, w, True
        if not improved:
            step *= 0.5
            if step < 1e-3:
                break
                
    return best_w / best_w[0], best_s

def train_nn(loss_type="weighted_ce", epochs=15, batch_size=2048, lr=0.003, weight_decay=1e-4, gamma=2.0, dry_run=False, save_prefix=None):
    print(f"--- Training Tabular MLP (Loss: {loss_type}, Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr}, Save Prefix: {save_prefix}) ---")
    
    # Load data
    train = pd.read_csv(ROOT / "data" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")
    folds = pd.read_csv(ROOT / "shared" / "folds.csv").set_index("id")["fold"]
    train["fold"] = train["id"].map(folds)
    
    # Target encoding
    target_map = {"at-risk": 0, "fit": 1, "unhealthy": 2}
    y_all = train["health_condition"].map(target_map).values
    
    # Preprocess categorical features (map unique values to 1..N, NaN maps to 0)
    cat_maps = {}
    for col in CAT:
        uniques = train[col].dropna().unique()
        cat_maps[col] = {val: idx + 1 for idx, val in enumerate(uniques)}
        
    print("Categorical encodings:")
    for col, mapping in cat_maps.items():
        print(f"  {col}: {mapping}")
        
    X_cat_train_full = np.zeros((len(train), len(CAT)), dtype=np.int32)
    X_cat_test = np.zeros((len(test), len(CAT)), dtype=np.int32)
    
    for i, col in enumerate(CAT):
        X_cat_train_full[:, i] = train[col].map(cat_maps[col]).fillna(0).values
        X_cat_test[:, i] = test[col].map(cat_maps[col]).fillna(0).values
        
    cat_dims = [len(cat_maps[col]) + 1 for col in CAT] # +1 for missing category (0)
    emb_dims = [4] * len(CAT)
    
    oof = np.zeros((len(train), 3))
    test_preds = np.zeros((len(test), 3))
    
    device = torch.device("cpu")
    
    num_folds = 1 if dry_run else 5
    print(f"Starting training on {num_folds} folds...")
    
    for k in range(num_folds):
        print(f"\n--- Fold {k} ---")
        tri = train["fold"].values != k
        vai = train["fold"].values == k
        
        # Numeric preprocessing
        scaler = StandardScaler()
        X_num_train = train.loc[tri, NUM].copy().values
        X_num_val = train.loc[vai, NUM].copy().values
        X_num_test = test[NUM].copy().values
        
        # Calculate missingness indicators
        na_train = np.isnan(X_num_train).astype(np.float32)
        na_val = np.isnan(X_num_val).astype(np.float32)
        na_test = np.isnan(X_num_test).astype(np.float32)
        
        # Impute with median
        medians = np.nanmedian(X_num_train, axis=0)
        for idx in range(len(NUM)):
            X_num_train[np.isnan(X_num_train[:, idx]), idx] = medians[idx]
            X_num_val[np.isnan(X_num_val[:, idx]), idx] = medians[idx]
            X_num_test[np.isnan(X_num_test[:, idx]), idx] = medians[idx]
            
        X_num_train_scaled = scaler.fit_transform(X_num_train)
        X_num_val_scaled = scaler.transform(X_num_val)
        X_num_test_scaled = scaler.transform(X_num_test)
        
        X_cont_train = np.hstack([X_num_train_scaled, na_train])
        X_cont_val = np.hstack([X_num_val_scaled, na_val])
        X_cont_test = np.hstack([X_num_test_scaled, na_test])
        
        X_cat_train = X_cat_train_full[tri]
        X_cat_val = X_cat_train_full[vai]
        
        y_train = y_all[tri]
        y_val = y_all[vai]
        
        # Convert training and val data to PyTorch tensors once
        t_x_cat_train = torch.tensor(X_cat_train, dtype=torch.long, device=device)
        t_x_cont_train = torch.tensor(X_cont_train, dtype=torch.float32, device=device)
        t_y_train = torch.tensor(y_train, dtype=torch.long, device=device)
        
        t_x_cat_val = torch.tensor(X_cat_val, dtype=torch.long, device=device)
        t_x_cont_val = torch.tensor(X_cont_val, dtype=torch.float32, device=device)
        
        t_x_cat_test = torch.tensor(X_cat_test, dtype=torch.long, device=device)
        t_x_cont_test = torch.tensor(X_cont_test, dtype=torch.float32, device=device)
        
        # Build model
        model = TabularMLP(cat_dims=cat_dims, emb_dims=emb_dims, num_features=X_cont_train.shape[1])
        model.to(device)
        
        loss_fn = get_loss_fn(loss_type, y_train, device, gamma=gamma)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_val_ba = 0.0
        best_model_state = None
        
        num_train_samples = len(y_train)
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            start_time = time.time()
            
            # Shuffle indices at the beginning of each epoch
            shuffled_indices = torch.randperm(num_train_samples, device=device)
            
            for i in range(0, num_train_samples, batch_size):
                batch_idx = shuffled_indices[i : i + batch_size]
                batch_cat = t_x_cat_train[batch_idx]
                batch_cont = t_x_cont_train[batch_idx]
                batch_y = t_y_train[batch_idx]
                
                optimizer.zero_grad()
                logits = model(batch_cat, batch_cont)
                loss = loss_fn(logits, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * len(batch_idx)
                
            scheduler.step()
            train_loss /= num_train_samples
            epoch_time = time.time() - start_time
            
            # Evaluate on validation using batch size 8192
            model.eval()
            val_preds = []
            with torch.no_grad():
                for i in range(0, len(y_val), 8192):
                    batch_cat = t_x_cat_val[i : i + 8192]
                    batch_cont = t_x_cont_val[i : i + 8192]
                    logits = model(batch_cat, batch_cont)
                    probs = F.softmax(logits, dim=1)
                    val_preds.append(probs.cpu().numpy())
            val_preds = np.concatenate(val_preds, axis=0)
            
            # Compute balanced accuracy
            val_labels = labels_from_proba(val_preds)
            val_ba = balanced_accuracy(np.asarray(CLASSES)[y_val], val_labels)
            
            # Track best model
            if val_ba > best_val_ba:
                best_val_ba = val_ba
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                
            print(f"  Epoch {epoch+1:2d}/{epochs:2d} | Train Loss: {train_loss:.4f} | Val Bal-Acc: {val_ba:.4f} | Best Val: {best_val_ba:.4f} | Time: {epoch_time:.1f}s")
            
            if dry_run and epoch >= 1:
                break
                
        # Load best model for final fold prediction
        model.load_state_dict(best_model_state)
        model.eval()
        
        # Final val prediction
        val_probs = []
        with torch.no_grad():
            for i in range(0, len(y_val), 8192):
                batch_cat = t_x_cat_val[i : i + 8192]
                batch_cont = t_x_cont_val[i : i + 8192]
                logits = model(batch_cat, batch_cont)
                val_probs.append(F.softmax(logits, dim=1).cpu().numpy())
        oof[vai] = np.concatenate(val_probs, axis=0)
        
        # Test predictions
        fold_test_probs = []
        with torch.no_grad():
            for i in range(0, len(test), 8192):
                batch_cat = t_x_cat_test[i : i + 8192]
                batch_cont = t_x_cont_test[i : i + 8192]
                logits = model(batch_cat, batch_cont)
                fold_test_probs.append(F.softmax(logits, dim=1).cpu().numpy())
        test_preds += np.concatenate(fold_test_probs, axis=0) / num_folds
        
        print(f"Fold {k} Best Val Balanced Accuracy: {best_val_ba:.4f}")
        
        if dry_run:
            break
            
    if not dry_run:
        # Save OOF and test probabilities
        y_labels = train["health_condition"].values
        ba_argmax = balanced_accuracy(y_labels, labels_from_proba(oof))
        w, ba_tuned = fast_tune_decision_weights(oof, y_labels)
        print(f"\nOverall OOF Balanced Accuracy (Argmax): {ba_argmax:.4f}")
        print(f"Overall OOF Balanced Accuracy (Tuned):  {ba_tuned:.4f} with weights {w.round(3)}")
        
        # Deliver csv files as per guidelines
        oof_path = ROOT / "neural_net" / f"{save_prefix}_oof.csv" if save_prefix else ROOT / "neural_net" / "oof.csv"
        test_path = ROOT / "neural_net" / f"{save_prefix}_test_proba.csv" if save_prefix else ROOT / "neural_net" / "test_proba.csv"
        
        pd.DataFrame({
            "id": train["id"],
            "p_at-risk": oof[:, 0],
            "p_fit": oof[:, 1],
            "p_unhealthy": oof[:, 2]
        }).to_csv(oof_path, index=False)
        
        pd.DataFrame({
            "id": test["id"],
            "p_at-risk": test_preds[:, 0],
            "p_fit": test_preds[:, 1],
            "p_unhealthy": test_preds[:, 2]
        }).to_csv(test_path, index=False)
        print(f"Wrote {oof_path.name} and {test_path.name}")
        
        return ba_argmax, ba_tuned, w, oof, test_preds
    else:
        print("Dry run completed successfully.")
        return None, None, None, None, None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", type=str, default="weighted_ce", choices=["ce", "weighted_ce", "focal", "weighted_focal"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--save_prefix", type=str, default=None)
    args = parser.parse_args()
    
    train_nn(
        loss_type=args.loss,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        gamma=args.gamma,
        dry_run=args.dry_run,
        save_prefix=args.save_prefix
    )
