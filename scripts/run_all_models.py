#!/usr/bin/env python
"""Train and evaluate ALL models on Evans dataset — full benchmark.

Includes:
  - Classical baselines (XGBoost, LightGBM, RF, kNN, MLP) on Morgan FP
  - Chemistry fingerprint models (DRFP, RXNFP) + classifiers
  - Generic transformers (DistilBERT, RoBERTa) on reaction SMILES
  - Chemistry-pretrained transformers (ChemBERTa-77M, MolT5-base) on reaction SMILES
"""

import json
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from aldolrxnmaster.evaluation.metrics import compute_all_metrics, compute_metrics_with_ci

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PROJECT = Path("/data2/zcwang/aldolrxnmaster")
FEAT_DIR = PROJECT / "data" / "processed" / "features"
SPLIT_DIR = PROJECT / "data" / "processed" / "splits"
RESULTS_DIR = PROJECT / "results"
for d in [RESULTS_DIR / "predictions", RESULTS_DIR / "tables"]:
    d.mkdir(parents=True, exist_ok=True)


def load_data_and_split(split_name):
    data = np.load(FEAT_DIR / "tabular_features.npz", allow_pickle=True)
    X_full = data["X"]
    feature_names = list(data["feature_names"])

    labels = pd.read_csv(FEAT_DIR / "labels.csv")
    y = labels["label_joint"].values.astype(int)

    with open(SPLIT_DIR / f"{split_name}.json") as f:
        split = json.load(f)

    tr, va, te = np.array(split["train"]), np.array(split["val"]), np.array(split["test"])

    # For GBDT: use descriptors + conditions + SVD-reduced FP
    # Layout: [FP_product(2048) | FP_rxn_diff(2048) | descriptors(~51) | conditions(variable)]
    fp_start = 0
    fp_end = 4096  # product FP + rxn_diff FP
    desc_start = 4096
    # Dynamically compute descriptor and condition dimensions
    n_desc = len([f for f in feature_names if f.startswith(("ketone_", "aldehyde_", "product_"))])
    desc_end = desc_start + n_desc
    cond_start = desc_end
    cond_end = X_full.shape[1]  # conditions go to the end

    X_fp = X_full[:, fp_start:fp_end]
    X_desc = X_full[:, desc_start:desc_end]
    X_cond = X_full[:, cond_start:cond_end]
    logger.info(f"  Features: FP={fp_end}d, desc={n_desc}d, cond={cond_end-cond_start}d, total={X_full.shape[1]}d")

    # SVD reduce fingerprints: 4096 -> 128
    svd = TruncatedSVD(n_components=128, random_state=42)
    svd.fit(X_fp[tr])
    X_fp_red = svd.transform(X_fp)

    X_tabular = np.hstack([X_fp_red, X_desc, X_cond])  # 128 + 51 + 14 = 193 features

    # For kNN: use raw product Morgan FP (first 2048)
    X_morgan = X_full[:, :2048]

    result = {
        "X_tab_train": X_tabular[tr], "X_tab_val": X_tabular[va], "X_tab_test": X_tabular[te],
        "X_morgan_train": X_morgan[tr], "X_morgan_val": X_morgan[va], "X_morgan_test": X_morgan[te],
        "X_full_train": X_full[tr], "X_full_val": X_full[va], "X_full_test": X_full[te],
        "y_train": y[tr], "y_val": y[va], "y_test": y[te],
        "train_idx": tr, "val_idx": va, "test_idx": te,
        "labels_df": labels, "svd": svd,
    }

    # Load precomputed chemistry fingerprints (DRFP + RXNFP)
    drfp_path = FEAT_DIR / "drfp_fps.npz"
    if drfp_path.exists():
        X_drfp = np.load(drfp_path)["X"].astype(np.float32)
        result.update({"X_drfp_train": X_drfp[tr], "X_drfp_val": X_drfp[va], "X_drfp_test": X_drfp[te]})
        # DRFP + reaction conditions hybrid
        X_drfp_cond = np.hstack([X_drfp, X_cond])
        result.update({"X_drfp_cond_train": X_drfp_cond[tr], "X_drfp_cond_val": X_drfp_cond[va], "X_drfp_cond_test": X_drfp_cond[te]})
        logger.info(f"  Loaded DRFP: {X_drfp.shape}")
    else:
        logger.warning("  DRFP not found — run precompute_chem_fps.py first")

    rxnfp_path = FEAT_DIR / "rxnfp_fps.npz"
    if rxnfp_path.exists():
        X_rxnfp = np.load(rxnfp_path)["X"].astype(np.float32)
        result.update({"X_rxnfp_train": X_rxnfp[tr], "X_rxnfp_val": X_rxnfp[va], "X_rxnfp_test": X_rxnfp[te]})
        logger.info(f"  Loaded RXNFP: {X_rxnfp.shape}")
    else:
        logger.warning("  RXNFP not found — run precompute_chem_fps.py first")

    return result


def load_rxn_smiles():
    return pd.read_csv(FEAT_DIR / "reaction_smiles.csv")["rxn_smiles_clean"].values


def evaluate_model(name, y_true, y_pred, y_prob=None, n_boot=500):
    m = compute_all_metrics(y_true, y_pred, y_prob)
    ci = compute_metrics_with_ci(y_true, y_pred, n_boot=n_boot)
    logger.info(f"  {name}: bal_acc={m['balanced_accuracy']:.4f}, MCC={m['mcc']:.4f}, "
                f"joint_acc={m['joint_accuracy']:.4f}, F1m={m['f1_macro']:.4f}")
    return {"name": name, "metrics": m, "ci": ci}


def save_preds(name, split_name, labels_df, test_idx, y_pred, y_prob=None):
    d = pd.DataFrame({"idx": test_idx, "y_true": labels_df.iloc[test_idx]["label_joint"].values.astype(int), "y_pred": y_pred})
    if y_prob is not None:
        for i in range(y_prob.shape[1]):
            d[f"prob_{i}"] = y_prob[:, i]
    d.to_csv(RESULTS_DIR / "predictions" / f"{name}_{split_name}.csv", index=False)


# ===================== MODELS =====================

def train_xgboost(data):
    import xgboost as xgb
    X_tr, y_tr = data["X_tab_train"], data["y_train"]
    sw = compute_sample_weight("balanced", y_tr)
    # Use well-tuned defaults instead of Optuna to save time
    configs = [
        {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.7},
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.6},
        {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.15, "subsample": 0.9, "colsample_bytree": 0.8},
    ]
    best_m, best_acc = None, 0
    for cfg in configs:
        cfg.update({"objective": "multi:softprob", "num_class": 4, "tree_method": "hist",
                    "random_state": 42, "n_jobs": -1, "verbosity": 0, "gamma": 0.1, "reg_lambda": 1.0})
        m = xgb.XGBClassifier(**cfg); m.fit(X_tr, y_tr, sample_weight=sw)
        acc = balanced_accuracy_score(data["y_val"], m.predict(data["X_tab_val"]))
        logger.info(f"  XGB config depth={cfg['max_depth']} trees={cfg['n_estimators']}: val_bacc={acc:.4f}")
        if acc > best_acc: best_acc, best_m = acc, m
    return best_m


def train_lightgbm(data):
    import lightgbm as lgb
    X_tr, y_tr = data["X_tab_train"], data["y_train"]
    configs = [
        {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1, "num_leaves": 31, "subsample": 0.8, "colsample_bytree": 0.7},
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05, "num_leaves": 47, "subsample": 0.8, "colsample_bytree": 0.6},
        {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.15, "num_leaves": 23, "subsample": 0.9, "colsample_bytree": 0.8},
    ]
    best_m, best_acc = None, 0
    for cfg in configs:
        cfg.update({"objective": "multiclass", "num_class": 4, "class_weight": "balanced",
                    "random_state": 42, "n_jobs": -1, "verbose": -1, "min_child_samples": 10})
        m = lgb.LGBMClassifier(**cfg); m.fit(X_tr, y_tr)
        acc = balanced_accuracy_score(data["y_val"], m.predict(data["X_tab_val"]))
        logger.info(f"  LGB config depth={cfg['max_depth']} trees={cfg['n_estimators']}: val_bacc={acc:.4f}")
        if acc > best_acc: best_acc, best_m = acc, m
    return best_m


def train_rf(data):
    m = RandomForestClassifier(n_estimators=500, max_depth=12, min_samples_split=5,
                               class_weight="balanced", random_state=42, n_jobs=-1)
    m.fit(data["X_tab_train"], data["y_train"])
    return m


def train_knn(data, k=5):
    m = KNeighborsClassifier(n_neighbors=k, metric="jaccard", n_jobs=-1)
    m.fit(data["X_morgan_train"], data["y_train"])
    return m


def train_mlp(data):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(data["X_tab_train"])
    m = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation="relu", solver="adam",
                      alpha=0.001, batch_size=64, learning_rate="adaptive", learning_rate_init=0.001,
                      max_iter=300, early_stopping=True, validation_fraction=0.15,
                      n_iter_no_change=20, random_state=42)
    m.fit(X_tr, data["y_train"])
    return m, scaler


def train_xgb_on(X_tr, y_tr, X_va, y_va):
    """Train XGBoost with 3-config grid on arbitrary features."""
    import xgboost as xgb
    sw = compute_sample_weight("balanced", y_tr)
    configs = [
        {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.7},
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.6},
        {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.15, "subsample": 0.9, "colsample_bytree": 0.8},
    ]
    best_m, best_acc = None, 0
    for cfg in configs:
        cfg.update({"objective": "multi:softprob", "num_class": 4, "tree_method": "hist",
                    "random_state": 42, "n_jobs": -1, "verbosity": 0, "gamma": 0.1, "reg_lambda": 1.0})
        m = xgb.XGBClassifier(**cfg); m.fit(X_tr, y_tr, sample_weight=sw)
        acc = balanced_accuracy_score(y_va, m.predict(X_va))
        if acc > best_acc: best_acc, best_m = acc, m
    logger.info(f"    best val_bacc={best_acc:.4f}")
    return best_m


def train_lgb_on(X_tr, y_tr, X_va, y_va):
    """Train LightGBM with 3-config grid on arbitrary features."""
    import lightgbm as lgb
    configs = [
        {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1, "num_leaves": 31, "subsample": 0.8, "colsample_bytree": 0.7},
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05, "num_leaves": 47, "subsample": 0.8, "colsample_bytree": 0.6},
        {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.15, "num_leaves": 23, "subsample": 0.9, "colsample_bytree": 0.8},
    ]
    best_m, best_acc = None, 0
    for cfg in configs:
        cfg.update({"objective": "multiclass", "num_class": 4, "class_weight": "balanced",
                    "random_state": 42, "n_jobs": -1, "verbose": -1, "min_child_samples": 10})
        m = lgb.LGBMClassifier(**cfg); m.fit(X_tr, y_tr)
        acc = balanced_accuracy_score(y_va, m.predict(X_va))
        if acc > best_acc: best_acc, best_m = acc, m
    logger.info(f"    best val_bacc={best_acc:.4f}")
    return best_m


def train_mlp_on(X_tr, y_tr):
    """Train MLP on arbitrary features with StandardScaler."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    m = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation="relu", solver="adam",
                      alpha=0.001, batch_size=64, learning_rate="adaptive", learning_rate_init=0.001,
                      max_iter=300, early_stopping=True, validation_fraction=0.15,
                      n_iter_no_change=20, random_state=42)
    m.fit(X_tr_s, y_tr)
    return m, scaler


def train_chemberta(data, rxn_smiles, epochs=15):
    """Fine-tune ChemBERTa-77M (chemistry-aware RoBERTa) for 4-class classification."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoTokenizer, AutoModel

    model_path = str(PROJECT / "external" / "pretrained_weights" / "DeepChem_ChemBERTa-77M-MLM")
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def encode(smi_arr, max_len=200):
        enc = tokenizer(list(smi_arr), padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"]

    tr_ids, tr_mask = encode(rxn_smiles[data["train_idx"]])
    va_ids, va_mask = encode(rxn_smiles[data["val_idx"]])

    tr_ds = TensorDataset(tr_ids, tr_mask, torch.tensor(data["y_train"], dtype=torch.long))
    va_ds = TensorDataset(va_ids, va_mask, torch.tensor(data["y_val"], dtype=torch.long))
    tr_dl = DataLoader(tr_ds, batch_size=32, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=64)

    class Clf(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = AutoModel.from_pretrained(model_path)
            h = self.enc.config.hidden_size  # 384
            self.head = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(h, 128),
                                            torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(128, 4))
        def forward(self, ids, mask):
            return self.head(self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0])

    # Try 3 learning rates, pick best on val
    best_model_state, best_overall_acc = None, 0
    best_tok = tokenizer
    for lr in [1e-5, 2e-5, 5e-5]:
        model = Clf().to(device)
        cw = compute_class_weight("balanced", classes=np.arange(4), y=data["y_train"])
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        crit = torch.nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(device))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        best_acc, best_state, patience = 0, None, 0
        for ep in range(epochs):
            model.train()
            for ids, mask, lab in tr_dl:
                ids, mask, lab = ids.to(device), mask.to(device), lab.to(device)
                opt.zero_grad(); loss = crit(model(ids, mask), lab); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            sched.step()

            model.eval(); preds = []
            with torch.no_grad():
                for ids, mask, _ in va_dl:
                    preds.extend(model(ids.to(device), mask.to(device)).argmax(1).cpu().numpy())
            acc = balanced_accuracy_score(data["y_val"], np.array(preds))
            if acc > best_acc:
                best_acc = acc; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; patience = 0
            else:
                patience += 1
            if patience >= 5: break

        logger.info(f"    ChemBERTa lr={lr}: best_val_bacc={best_acc:.4f}")
        if best_acc > best_overall_acc:
            best_overall_acc = best_acc; best_model_state = best_state
        del model; torch.cuda.empty_cache()

    # Rebuild best model
    final_model = Clf().to(device)
    final_model.load_state_dict(best_model_state)
    return final_model, tokenizer, device


def train_molt5(data, rxn_smiles, epochs=15):
    """Fine-tune MolT5-base encoder for 4-class classification."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoTokenizer, T5EncoderModel

    model_path = str(PROJECT / "external" / "pretrained_weights" / "laituan245_molt5-base")
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def encode(smi_arr, max_len=200):
        enc = tokenizer(list(smi_arr), padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"]

    tr_ids, tr_mask = encode(rxn_smiles[data["train_idx"]])
    va_ids, va_mask = encode(rxn_smiles[data["val_idx"]])

    tr_ds = TensorDataset(tr_ids, tr_mask, torch.tensor(data["y_train"], dtype=torch.long))
    va_ds = TensorDataset(va_ids, va_mask, torch.tensor(data["y_val"], dtype=torch.long))
    tr_dl = DataLoader(tr_ds, batch_size=8, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=16)

    class MolT5Clf(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = T5EncoderModel.from_pretrained(model_path)
            self.enc.gradient_checkpointing_enable()
            # Freeze embedding + first 6 of 12 encoder layers (keep 6 trainable)
            for p in self.enc.encoder.embed_tokens.parameters():
                p.requires_grad = False
            for i, layer in enumerate(self.enc.encoder.block):
                if i < 6:
                    for p in layer.parameters():
                        p.requires_grad = False
            h = self.enc.config.d_model  # 768
            self.head = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(h, 256),
                                            torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(256, 4))

        def forward(self, ids, mask):
            out = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state  # (B, L, 768)
            # Mean-pooling over non-padding tokens
            mask_exp = mask.unsqueeze(-1).float()  # (B, L, 1)
            pooled = (out * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1)  # (B, 768)
            return self.head(pooled)

    model = MolT5Clf().to(device)
    cw = compute_class_weight("balanced", classes=np.arange(4), y=data["y_train"])
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-5, weight_decay=0.01)
    crit = torch.nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(device))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_acc, best_state, patience = 0, None, 0
    accum_steps = 4  # effective batch = 8 * 4 = 32
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        for step, (ids, mask, lab) in enumerate(tr_dl):
            ids, mask, lab = ids.to(device), mask.to(device), lab.to(device)
            loss = crit(model(ids, mask), lab) / accum_steps
            loss.backward()
            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad()
        if (step + 1) % accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad()
        sched.step()

        model.eval(); preds = []
        with torch.no_grad():
            for ids, mask, _ in va_dl:
                preds.extend(model(ids.to(device), mask.to(device)).argmax(1).cpu().numpy())
        acc = balanced_accuracy_score(data["y_val"], np.array(preds))
        logger.info(f"    MolT5 ep{ep}: val_bacc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; patience = 0
        else:
            patience += 1
        if patience >= 6: logger.info(f"    Early stop ep{ep}"); break

    model.load_state_dict(best_state); model.to(device)
    return model, tokenizer, device


def predict_molt5(model, tok, dev, smiles, bs=16):
    """Predict with MolT5 encoder model."""
    import torch
    model.eval(); all_p = []
    for i in range(0, len(smiles), bs):
        enc = tok(list(smiles[i:i+bs]), padding="max_length", truncation=True, max_length=200, return_tensors="pt")
        with torch.no_grad():
            logits = model(enc["input_ids"].to(dev), enc["attention_mask"].to(dev))
            all_p.append(torch.softmax(logits, 1).cpu().numpy())
    p = np.vstack(all_p); return p.argmax(1), p


def train_transformer(data, rxn_smiles, model_name="distilbert-base-uncased", epochs=15):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoTokenizer, AutoModel

    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(smi_arr, max_len=200):
        enc = tokenizer(list(smi_arr), padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"]

    tr_ids, tr_mask = encode(rxn_smiles[data["train_idx"]])
    va_ids, va_mask = encode(rxn_smiles[data["val_idx"]])

    tr_ds = TensorDataset(tr_ids, tr_mask, torch.tensor(data["y_train"], dtype=torch.long))
    va_ds = TensorDataset(va_ids, va_mask, torch.tensor(data["y_val"], dtype=torch.long))
    tr_dl = DataLoader(tr_ds, batch_size=32, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=64)

    class Clf(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = AutoModel.from_pretrained(model_name)
            h = self.enc.config.hidden_size
            self.head = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(h, 128),
                                            torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(128, 4))
        def forward(self, ids, mask):
            return self.head(self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0])

    model = Clf().to(device)
    cw = compute_class_weight("balanced", classes=np.arange(4), y=data["y_train"])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    crit = torch.nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(device))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_acc, best_state, patience = 0, None, 0
    for ep in range(epochs):
        model.train()
        for ids, mask, lab in tr_dl:
            ids, mask, lab = ids.to(device), mask.to(device), lab.to(device)
            opt.zero_grad(); loss = crit(model(ids, mask), lab); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()

        model.eval(); preds = []
        with torch.no_grad():
            for ids, mask, _ in va_dl:
                preds.extend(model(ids.to(device), mask.to(device)).argmax(1).cpu().numpy())
        acc = balanced_accuracy_score(data["y_val"], np.array(preds))
        if acc > best_acc:
            best_acc = acc; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; patience = 0
        else:
            patience += 1
        if ep % 3 == 0: logger.info(f"    ep{ep}: val_bacc={acc:.4f}")
        if patience >= 5: logger.info(f"    Early stop ep{ep}"); break

    model.load_state_dict(best_state); model.to(device)
    return model, tokenizer, device


def predict_tf(model, tok, dev, smiles, bs=64):
    import torch
    model.eval(); all_p = []
    for i in range(0, len(smiles), bs):
        enc = tok(list(smiles[i:i+bs]), padding="max_length", truncation=True, max_length=200, return_tensors="pt")
        with torch.no_grad():
            all_p.append(torch.softmax(model(enc["input_ids"].to(dev), enc["attention_mask"].to(dev)), 1).cpu().numpy())
    p = np.vstack(all_p); return p.argmax(1), p


# ===================== MAIN =====================

def run_split(split_name):
    logger.info(f"\n{'='*60}\n  SPLIT: {split_name}\n{'='*60}")
    data = load_data_and_split(split_name)
    rxn_smi = load_rxn_smiles()

    logger.info(f"Train={len(data['y_train'])}, Val={len(data['y_val'])}, Test={len(data['y_test'])}")
    logger.info(f"Train classes: {np.bincount(data['y_train'], minlength=4)}")
    logger.info(f"Test classes:  {np.bincount(data['y_test'], minlength=4)}")

    results = []

    def run_model(name, pred_fn):
        y_pred, y_prob = pred_fn()
        r = evaluate_model(name, data["y_test"], y_pred, y_prob)
        save_preds(name.lower().replace(" ", "_").replace("-", "_"), split_name, data["labels_df"], data["test_idx"], y_pred, y_prob)
        results.append(r)

    # --- Baselines ---
    logger.info("\n--- XGBoost ---")
    xgb_m = train_xgboost(data)
    run_model("XGBoost", lambda: (xgb_m.predict(data["X_tab_test"]), xgb_m.predict_proba(data["X_tab_test"])))

    logger.info("\n--- LightGBM ---")
    lgb_m = train_lightgbm(data)
    run_model("LightGBM", lambda: (lgb_m.predict(data["X_tab_test"]), lgb_m.predict_proba(data["X_tab_test"])))

    # XGBoost on FULL features (no SVD) for comparison
    logger.info("\n--- XGBoost-FullFP ---")
    import xgboost as xgb
    sw = compute_sample_weight("balanced", data["y_train"])
    xgb_full = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8,
                                  colsample_bytree=0.3, objective="multi:softprob", num_class=4,
                                  tree_method="hist", random_state=42, n_jobs=-1, verbosity=0)
    xgb_full.fit(data["X_full_train"], data["y_train"], sample_weight=sw)
    run_model("XGBoost-FullFP", lambda: (xgb_full.predict(data["X_full_test"]), xgb_full.predict_proba(data["X_full_test"])))

    logger.info("\n--- Random Forest ---")
    rf_m = train_rf(data)
    run_model("RF", lambda: (rf_m.predict(data["X_tab_test"]), rf_m.predict_proba(data["X_tab_test"])))

    logger.info("\n--- 1-NN ---")
    knn1 = train_knn(data, 1)
    run_model("1-NN", lambda: (knn1.predict(data["X_morgan_test"]), knn1.predict_proba(data["X_morgan_test"])))

    logger.info("\n--- 5-NN ---")
    knn5 = train_knn(data, 5)
    run_model("5-NN", lambda: (knn5.predict(data["X_morgan_test"]), knn5.predict_proba(data["X_morgan_test"])))

    logger.info("\n--- Morgan-MLP ---")
    mlp_m, mlp_sc = train_mlp(data)
    Xt = mlp_sc.transform(data["X_tab_test"])
    run_model("Morgan-MLP", lambda: (mlp_m.predict(Xt), mlp_m.predict_proba(Xt)))

    # --- Chemistry Fingerprint Models ---
    if "X_drfp_train" in data:
        logger.info("\n--- DRFP+XGBoost ---")
        m = train_xgb_on(data["X_drfp_train"], data["y_train"], data["X_drfp_val"], data["y_val"])
        run_model("DRFP+XGBoost", lambda: (m.predict(data["X_drfp_test"]), m.predict_proba(data["X_drfp_test"])))

        logger.info("\n--- DRFP+LightGBM ---")
        m = train_lgb_on(data["X_drfp_train"], data["y_train"], data["X_drfp_val"], data["y_val"])
        run_model("DRFP+LightGBM", lambda: (m.predict(data["X_drfp_test"]), m.predict_proba(data["X_drfp_test"])))

        logger.info("\n--- DRFP+Cond+XGBoost ---")
        m = train_xgb_on(data["X_drfp_cond_train"], data["y_train"], data["X_drfp_cond_val"], data["y_val"])
        run_model("DRFP+Cond+XGBoost", lambda: (m.predict(data["X_drfp_cond_test"]), m.predict_proba(data["X_drfp_cond_test"])))

    if "X_rxnfp_train" in data:
        logger.info("\n--- RXNFP+XGBoost ---")
        m = train_xgb_on(data["X_rxnfp_train"], data["y_train"], data["X_rxnfp_val"], data["y_val"])
        run_model("RXNFP+XGBoost", lambda: (m.predict(data["X_rxnfp_test"]), m.predict_proba(data["X_rxnfp_test"])))

        logger.info("\n--- RXNFP+LightGBM ---")
        m = train_lgb_on(data["X_rxnfp_train"], data["y_train"], data["X_rxnfp_val"], data["y_val"])
        run_model("RXNFP+LightGBM", lambda: (m.predict(data["X_rxnfp_test"]), m.predict_proba(data["X_rxnfp_test"])))

        logger.info("\n--- RXNFP+MLP ---")
        m, sc = train_mlp_on(data["X_rxnfp_train"], data["y_train"])
        Xt_rxnfp = sc.transform(data["X_rxnfp_test"])
        run_model("RXNFP+MLP", lambda: (m.predict(Xt_rxnfp), m.predict_proba(Xt_rxnfp)))

    # --- Transformers ---
    logger.info("\n--- DistilBERT-RxnSMILES ---")
    try:
        tm, tt, td = train_transformer(data, rxn_smi, "distilbert-base-uncased", 15)
        ts = rxn_smi[data["test_idx"]]
        run_model("DistilBERT-Rxn", lambda: predict_tf(tm, tt, td, ts))
        del tm; import torch; torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"DistilBERT failed: {e}")

    logger.info("\n--- RoBERTa-RxnSMILES ---")
    try:
        tm, tt, td = train_transformer(data, rxn_smi, "roberta-base", 15)
        ts = rxn_smi[data["test_idx"]]
        run_model("RoBERTa-Rxn", lambda: predict_tf(tm, tt, td, ts))
        del tm; import torch; torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"RoBERTa failed: {e}")

    # --- Chemistry-Pretrained Transformers ---
    logger.info("\n--- ChemBERTa-77M ---")
    try:
        tm, tt, td = train_chemberta(data, rxn_smi, epochs=15)
        ts = rxn_smi[data["test_idx"]]
        run_model("ChemBERTa-77M", lambda: predict_tf(tm, tt, td, ts))
        del tm; import torch; torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"ChemBERTa failed: {e}")

    logger.info("\n--- MolT5-base ---")
    try:
        tm, tt, td = train_molt5(data, rxn_smi, epochs=10)
        ts = rxn_smi[data["test_idx"]]
        run_model("MolT5-base", lambda: predict_molt5(tm, tt, td, ts))
        del tm; import torch; torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"MolT5 failed: {e}")

    # --- Baselines ---
    logger.info("\n--- Majority Class ---")
    mc = np.bincount(data["y_train"]).argmax()
    yp = np.full(len(data["y_test"]), mc)
    pp = np.zeros((len(data["y_test"]), 4)); pp[:, mc] = 1.0
    results.append(evaluate_model("MajorityClass", data["y_test"], yp, pp))
    save_preds("majorityclass", split_name, data["labels_df"], data["test_idx"], yp, pp)

    logger.info("\n--- Random ---")
    rng = np.random.RandomState(42)
    yp = rng.randint(0, 4, len(data["y_test"]))
    pp = np.zeros((len(data["y_test"]), 4))
    for i in range(len(yp)):
        pp[i] = rng.dirichlet([1, 1, 1, 1])
    results.append(evaluate_model("Random", data["y_test"], yp, pp))
    save_preds("random", split_name, data["labels_df"], data["test_idx"], yp, pp)

    # --- Results Table ---
    rows = []
    for r in results:
        m = r["metrics"]; ci = r.get("ci", {})
        row = {"Model": r["name"],
               "Bal.Acc": f"{m['balanced_accuracy']:.4f}",
               "MCC": f"{m['mcc']:.4f}",
               "Joint": f"{m['joint_accuracy']:.4f}",
               "F1m": f"{m['f1_macro']:.4f}",
               "Ca": f"{m['ca_accuracy']:.4f}",
               "Cb": f"{m['cb_accuracy']:.4f}",
               "SA": f"{m['sa_accuracy']:.4f}",
               "F1_C0": f"{m['f1_class0']:.3f}",
               "F1_C1": f"{m['f1_class1']:.3f}",
               "F1_C2": f"{m['f1_class2']:.3f}",
               "F1_C3": f"{m['f1_class3']:.3f}"}
        if "balanced_accuracy" in ci:
            ba = ci["balanced_accuracy"]
            row["Bal.Acc 95%CI"] = f"[{ba['ci_lo']:.3f},{ba['ci_hi']:.3f}]"
        rows.append(row)

    rdf = pd.DataFrame(rows)
    print(f"\n{rdf.to_string(index=False)}")
    rdf.to_csv(RESULTS_DIR / "tables" / f"comparison_{split_name}.csv", index=False)

    with open(RESULTS_DIR / "tables" / f"full_{split_name}.json", "w") as f:
        json.dump([{"name": r["name"], "metrics": {k: v for k, v in r["metrics"].items() if k != "confusion_matrix"},
                    "cm": r["metrics"].get("confusion_matrix"), "ci": r.get("ci")} for r in results], f, indent=2, default=str)

    return results


if __name__ == "__main__":
    # Run temporal first (primary evaluation), then others
    run_split("evans_temporal")
    run_split("evans_scaffold")
    run_split("evans_grouped_random_seed42")
    logger.info("\nDone! All results in /data2/zcwang/aldolrxnmaster/results/tables/")
