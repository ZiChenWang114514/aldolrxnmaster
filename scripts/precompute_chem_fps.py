#!/usr/bin/env python
"""Precompute chemistry-specific reaction fingerprints (DRFP + RXNFP).

Must run BEFORE run_all_models.py so fingerprint-based models can load cached features.
Uses aldol-rxn conda environment.

Output:
  data/processed/features/drfp_fps.npz   (1822 x 2048, binary)
  data/processed/features/rxnfp_fps.npz  (1822 x 256, float32)
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PROJECT = Path("/data2/zcwang/aldolrxnmaster")
FEAT_DIR = PROJECT / "data" / "processed" / "features"
EXTERNAL = PROJECT / "external"


def compute_drfp(rxn_smiles: list[str]) -> np.ndarray:
    """Compute DRFP fingerprints for reaction SMILES."""
    sys.path.insert(0, str(EXTERNAL / "drfp" / "src"))
    from drfp import DrfpEncoder

    logger.info(f"Computing DRFP for {len(rxn_smiles)} reactions...")
    t0 = time.time()

    n_bits = 2048
    fps = []
    n_fail = 0
    for i, smi in enumerate(rxn_smiles):
        try:
            fp = DrfpEncoder.encode(
                [smi],
                n_folded_length=n_bits,
                radius=3,
                rings=True,
            )
            fps.append(fp[0])
        except Exception:
            fps.append(np.zeros(n_bits, dtype=np.uint8))
            n_fail += 1
        if (i + 1) % 500 == 0:
            logger.info(f"  DRFP progress: {i+1}/{len(rxn_smiles)}")

    X = np.array(fps, dtype=np.int8)
    logger.info(f"  DRFP done: shape={X.shape}, density={X.mean():.4f}, "
                f"failures={n_fail}, time={time.time()-t0:.1f}s")
    return X


def compute_rxnfp(rxn_smiles: list[str]) -> np.ndarray:
    """Compute RXNFP fingerprints using bundled BERT model.

    Uses direct model inference since rxnfp's tokenizer API is incompatible
    with transformers>=5.0 (batch_encode_plus removed from SmilesTokenizer).
    """
    import torch
    from transformers import BertModel

    model_dir = str(EXTERNAL / "rxnfp" / "rxnfp" / "models" / "transformers" / "bert_ft")

    logger.info(f"Computing RXNFP for {len(rxn_smiles)} reactions...")
    t0 = time.time()

    # Load BERT model directly
    model = BertModel.from_pretrained(model_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Load the SmilesTokenizer from rxnfp
    sys.path.insert(0, str(EXTERNAL / "rxnfp"))
    from rxnfp.tokenization import SmilesTokenizer
    tokenizer = SmilesTokenizer(vocab_file=str(Path(model_dir) / "vocab.txt"))

    max_len = model.config.max_position_embeddings  # 512

    fps = []
    n_fail = 0
    batch_size = 16
    for i in range(0, len(rxn_smiles), batch_size):
        batch = rxn_smiles[i:i+batch_size]
        try:
            # Tokenize one by one and pad manually
            encodings = []
            for smi in batch:
                enc = tokenizer.encode(str(smi), max_length=max_len, truncation=True)
                encodings.append(enc)

            # Pad to max length in batch
            max_batch_len = max(len(e) for e in encodings)
            input_ids = torch.zeros(len(batch), max_batch_len, dtype=torch.long)
            attention_mask = torch.zeros(len(batch), max_batch_len, dtype=torch.long)
            for j, enc in enumerate(encodings):
                input_ids[j, :len(enc)] = torch.tensor(enc)
                attention_mask[j, :len(enc)] = 1

            with torch.no_grad():
                output = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
            # [CLS] token at position 0
            cls_emb = output.last_hidden_state[:, 0, :].cpu().numpy()
            fps.extend(cls_emb)
        except Exception as e:
            # Fallback: zero vector for failed batch
            for _ in batch:
                fps.append(np.zeros(model.config.hidden_size, dtype=np.float32))
                n_fail += 1

        if (i + batch_size) % 500 < batch_size:
            logger.info(f"  RXNFP progress: {min(i+batch_size, len(rxn_smiles))}/{len(rxn_smiles)}")

    X = np.array(fps, dtype=np.float32)
    logger.info(f"  RXNFP done: shape={X.shape}, failures={n_fail}, time={time.time()-t0:.1f}s")
    return X


def main():
    # Load reaction SMILES
    rxn_df = pd.read_csv(FEAT_DIR / "reaction_smiles.csv")
    rxn_smiles = rxn_df["rxn_smiles_clean"].tolist()
    logger.info(f"Loaded {len(rxn_smiles)} reaction SMILES")

    # Filter out empty SMILES
    valid_mask = [bool(s and ">>" in str(s)) for s in rxn_smiles]
    n_valid = sum(valid_mask)
    logger.info(f"Valid reaction SMILES: {n_valid}/{len(rxn_smiles)}")

    # DRFP
    X_drfp = compute_drfp(rxn_smiles)
    np.savez_compressed(FEAT_DIR / "drfp_fps.npz", X=X_drfp)
    logger.info(f"Saved DRFP: {FEAT_DIR / 'drfp_fps.npz'}")

    # RXNFP
    X_rxnfp = compute_rxnfp(rxn_smiles)
    np.savez_compressed(FEAT_DIR / "rxnfp_fps.npz", X=X_rxnfp)
    logger.info(f"Saved RXNFP: {FEAT_DIR / 'rxnfp_fps.npz'}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
