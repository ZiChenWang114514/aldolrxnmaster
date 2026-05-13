# Reproduction Guide

在任意机器上从零复现 AldolRxnMaster 全部结果。

## 前置条件

- Linux, CUDA GPU (RTX 4090 或同等)
- Conda (Anaconda/Miniconda)
- 项目代码 (`git clone` 或拷贝本目录)

---

## Step 0: 创建 Conda 环境

```bash
# 主环境
conda create -n aldol-rxn python=3.11 -y
conda activate aldol-rxn

# PyTorch + CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 核心依赖
pip install rdkit xgboost lightgbm scikit-learn pandas numpy scipy
pip install transformers lightning optuna
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.11.0+cu128.html

# 化学专用
pip install chemprop
pip install "httpx[socks]"

# 外部包 (editable installs)
cd external/drfp && pip install -e . && cd ../..
pip install rxnfp --no-deps

# EquiReact 单独环境 (可选, 用于 3D 模型)
conda create -n equireact python=3.11 -y
conda run -n equireact pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
conda run -n equireact pip install torch-geometric e3nn scikit-learn pandas numpy scipy rdkit tqdm pyaml
conda run -n equireact pip install pyg-lib torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.11.0+cu128.html
```

## Step 1: 数据清洗 (已完成, 可跳过)

```bash
conda activate aldol-rxn
cd /path/to/aldolrxnmaster

# 从原始数据生成 interim + processed
python scripts/03_run_cleaning.py
# 输出: data/interim/01-05_*.csv, data/processed/evans_clean.csv, data/processed/splits/*.json
```

## Step 2: 特征工程

```bash
python -c "
import logging, sys
from pathlib import Path
logging.basicConfig(level=logging.INFO)
sys.path.insert(0, 'src')
from aldolrxnmaster.features.compute_all import run
run(Path('.'))
"
```

**输出** (`data/processed/features/`):
- `labels.csv` — 4-class joint labels (Ca×Cb)
- `reaction_smiles.csv` — clean reaction SMILES
- `reaction_conditions.csv` — **35 维**: metal (9d) + solvent Kamlet-Taft (5d) + reagent/base (21d)
- `rdkit_descriptors.csv` — 51 维 RDKit 2D descriptors
- `tabular_features.npz` — 4182 维合并特征 (FP 4096 + desc 51 + cond 35)
- `morgan_fps.npz` — Morgan FP 分离存储
- `auxchiral_features.csv` — Evans auxiliary chirality features (6d)

### Reagent/Base 编码详情 (21 维)

`compute_all.py` 中的 `compute_reagent_features()` 按化学角色分类:

**碱 (base_type, 8 维 one-hot)**:
| 类别 | 对应试剂 | Evans 数据中的数量 |
|---|---|---|
| DIPEA | n-ethyl-n,n-diisopropylamine (Hünig's base) | 767 |
| Et3N | triethylamine | 607 |
| LiHMDS | lithium hexamethyldisilazane | 79 |
| NaHMDS | sodium hexamethyldisilazane | 0 |
| LDA | lithium diisopropylamide | 0 |
| KHMDS | potassium hexamethyldisilazane | 0 |
| other_base | 2,6-lutidine, pyridine, DMAP, DBU, sparteine 等 | 0 |
| no_base | 无碱或未知 | 369 |

**活化剂 (activator_type, 9 维 one-hot)**:
| 类别 | 对应试剂 |
|---|---|
| Bu2BOTf | di-n-butylboryl trifluoromethanesulfonate |
| Chx2BCl | dicyclohexylboron chloride |
| Ipc2BCl | diisopinocampheylboron chloride |
| TiCl4 | titanium tetrachloride |
| Sn_OTf2 | tin(ii) trifluoromethanesulfonate |
| MgCl2 | magnesium chloride / bromide |
| BF3_OEt2 | boron trifluoride etherate |
| other_activator | 其他活化剂 |
| no_activator | 无活化剂 |

**其他 (4 维 binary)**:
- `has_oxidant`: H₂O₂ 等氧化剂
- `has_silylating`: TMSCl 等硅基化试剂
- `has_additive`: TMEDA, DMPU, HMPA 等添加剂
- `reagent_known`: 是否有已知试剂信息

### Evans Auxiliary Chirality 编码详情 (6 维)

`compute_all.py` 中的 `compute_auxiliary_chirality()`:
- `aux_config_R`: 主手性中心 R=1, S=0, 未知=-1
- `aux_n_stereocenters`: 辅助基上定义的手性中心数
- `aux_has_benzyl`: 是否有苄基取代
- `aux_has_isopropyl`: 是否有异丙基取代
- `aux_has_phenyl`: 是否有苯基直接连接环 C
- `aux_mw`: 辅助基分子量

## Step 3: 预计算化学指纹

```bash
python scripts/precompute_chem_fps.py
# 输出: data/processed/features/drfp_fps.npz (1822×2048), rxnfp_fps.npz (1822×256)
```

## Step 4: 生成 3D Conformers (可选, ChiENN/EquiReact 需要)

```bash
python scripts/generate_3d_conformers.py
# 输出: data/processed/conformers/conformers.pkl (~17MB, 1801/1822 valid)
```

## Step 5: 跑全部模型

```bash
# 主 benchmark: 17+ baseline 模型 (XGBoost, LightGBM, RF, kNN, MLP,
#   DRFP×3, RXNFP×3, DistilBERT, RoBERTa, ChemBERTa, MolT5, AuxChiral 消融)
python scripts/run_all_models.py

# Phase 6 化学 SOTA 模型 (可并行)
python scripts/run_chemprop.py          # Chemprop MPNN ± conditions
python scripts/run_protonet.py          # Prototypical Networks
python scripts/run_chemahnet.py         # ChemAHNet-style chemistry-informed DL
python scripts/run_chienn_product.py    # ChiENN chirality-aware GNN (需 3D conformers)

# EquiReact (需 equireact 环境)
conda run -n equireact python scripts/run_equireact.py
```

## Step 6: 重建统一对比表

```bash
python scripts/rebuild_comparison.py
# 输出: results/tables/comparison_evans_{temporal,scaffold,grouped_random_seed42}.csv
```

---

## 修改过的关键源文件

### `src/aldolrxnmaster/features/compute_all.py`

Phase 7 修改 (reagent/base + auxiliary chirality 编码):
- 新增 `_parse_reagent_list()` — 解析 Reagents 列的 Python list string
- 新增 `BASE_MAP`, `ACTIVATOR_MAP`, `OXIDANT_KEYWORDS`, `SILYLATING_KEYWORDS`, `ADDITIVE_MAP` — 90 种试剂到角色的映射字典
- 新增 `compute_reagent_features()` — 按角色分类编码, 输出 21 维
- 新增 `compute_auxiliary_chirality()` — Evans 辅助基手性特征, 输出 6 维
- 修改 `run()` — 条件特征从 `[metal, solvent]` 变为 `[metal, solvent, reagent]`, 并新增 auxchiral 输出

### `scripts/run_all_models.py`

- `load_data_and_split()` 中 `cond_end` 从硬编码 `cond_start + 14` 改为动态 `X_full.shape[1]`
- 新增 AuxChiral 消融模型 (由用户修改添加)

### `scripts/rebuild_comparison.py`

- `NAME_MAP` 新增: `auxchiral_xgboost`, `auxchiral_ald_xgboost`, `auxchiral_lgbm`, `auxchiral_noaux_xgboost`, `auxchiral_nobase_xgboost`
- `MODEL_ORDER` 更新排序

### `scripts/run_chemahnet.py`

- conditions 拆分: 从 `metal_cols + solvent_cols` 改为 `metal_cols + solvent_cols + reagent_cols`

---

## 预期结果 (Phase 5 基准, reagent 编码前)

| Rank | Model | Temporal Bal.Acc |
|------|-------|-----------------|
| 1 | DRFP+Cond+XGBoost | 0.7056 |
| 2 | Chemprop+Cond | 0.6327 |
| 3 | DistilBERT-Rxn | 0.5910 |
| 4 | DRFP+XGBoost | 0.5870 |
| 5 | ProtoNet | 0.5707 |

加入 reagent/base 编码后, 使用 conditions 的模型 (DRFP+Cond, Chemprop+Cond 等) 预期会有提升。
