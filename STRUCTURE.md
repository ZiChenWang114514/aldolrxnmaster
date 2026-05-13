# Project Structure

```
aldolrxnmaster/
├── CLAUDE.md                          # Claude Code 项目指引
├── RESULTS.md                         # 23 模型 benchmark 结果
├── STRUCTURE.md                       # 本文件
├── TODO.md                            # 待做事项
├── ISSUES.md                          # 已知问题与缺陷
├── CHANGELOG.md                       # 版本变更记录
├── pyproject.toml                     # 项目元数据
├── environment.yml                    # aldol-rxn conda 环境规格
│
├── data/
│   ├── raw/                           # 原始数据 (只读, 5 CSVs)
│   │   ├── alldata.csv                # 4751 行全量 aldol 反应数据
│   │   ├── evans_aux.csv              # Evans auxiliary 子集
│   │   ├── evans_aux_mapped.csv       # 带 atom mapping 版本
│   │   ├── evans_aux_reaction_train.csv
│   │   └── evans_aux_reactants_train.csv
│   │
│   ├── interim/                       # 7 步清洗中间产物
│   │   ├── 01_consolidated.csv        # Step 1: 合并 (4751 行)
│   │   ├── 02_deduplicated.csv        # Step 2: 去重 (4447 行)
│   │   ├── 03_validated.csv           # Step 3: SMILES 校验
│   │   ├── 04_labels_unified.csv      # Step 4: 4-class 统一标签
│   │   ├── 05_imputed.csv             # Step 5: 缺失值处理
│   │   └── *.log                      # 每步审计日志
│   │
│   ├── processed/
│   │   ├── evans_clean.csv            # 最终 Evans 清洗数据 (1822 行, 39 列)
│   │   ├── features/
│   │   │   ├── labels.csv             # 标签 (label_Ca, Cb, SA, joint, group_id, Year)
│   │   │   ├── reaction_smiles.csv    # 清洗后反应 SMILES (reactants>>product)
│   │   │   ├── reaction_conditions.csv # 反应条件 (metal 9d + solvent 5d = 14d)
│   │   │   ├── rdkit_descriptors.csv  # RDKit 2D 描述符 (51d)
│   │   │   ├── morgan_fps.npz         # Morgan FP (ketone/aldehyde/product/rxn_diff, 各 2048-bit)
│   │   │   ├── tabular_features.npz   # 合并表格特征 (4161d)
│   │   │   ├── drfp_fps.npz           # DRFP 反应指纹 (1822×2048, binary)
│   │   │   └── rxnfp_fps.npz          # RXNFP 反应指纹 (1822×256, float)
│   │   ├── splits/
│   │   │   ├── evans_temporal.json     # 时间分割 (train≤2015, val 2016-18, test≥2019)
│   │   │   ├── evans_scaffold.json     # Murcko scaffold 分割
│   │   │   └── evans_grouped_random_seed*.json  # 分组随机分割 (5 seeds)
│   │   └── conformers/
│   │       └── conformers.pkl          # 3D 构象 (ketone/aldehyde/product, ETKDG+MMFF)
│   │
│   └── quality_report/
│       ├── report.txt                  # 数据质量文本报告
│       └── report.json                 # 数据质量 JSON
│
├── src/aldolrxnmaster/
│   ├── data/                           # 数据清洗 pipeline (7 modules)
│   │   ├── consolidate.py              # Step 1: 数据合并
│   │   ├── deduplicate.py              # Step 2: 去重 + group_id
│   │   ├── validate_smiles.py          # Step 3: SMILES 校验
│   │   ├── unify_labels.py             # Step 4: 标签统一
│   │   ├── impute.py                   # Step 5: 缺失值填充 (Kamlet-Taft)
│   │   ├── split.py                    # Step 6: 数据分割
│   │   └── quality_report.py           # Step 7: 质量报告
│   ├── features/
│   │   └── compute_all.py              # 特征工程 (Morgan+RDKit+Conditions)
│   ├── evaluation/
│   │   └── metrics.py                  # 评估指标 (bal_acc, MCC, F1, CI)
│   ├── models/                         # 模型目录 (placeholder)
│   ├── training/                       # 训练工具 (placeholder)
│   └── utils/
│
├── scripts/
│   ├── 03_run_cleaning.py              # 数据清洗 orchestrator (Steps 1-7)
│   ├── precompute_chem_fps.py          # DRFP + RXNFP 指纹预计算
│   ├── run_all_models.py               # 主 benchmark (17 models: baselines+FP+transformers)
│   ├── run_chemprop.py                 # Chemprop MPNN ± conditions
│   ├── run_protonet.py                 # Prototypical Networks (meta-learning)
│   ├── run_chemahnet.py                # ChemAHNet-style chemistry-informed DL
│   ├── run_chienn_product.py           # ChiENN chirality-aware GNN
│   ├── run_equireact.py                # EquiReact 3D equivariant (equireact env)
│   ├── generate_3d_conformers.py       # 3D 构象生成 (RDKit ETKDG+MMFF)
│   ├── rebuild_comparison.py           # 从 prediction CSVs 重建统一对比表
│   ├── rerun_failed_models.py          # 重跑失败模型 (DistilBERT/RoBERTa/MolT5)
│   └── run_t5chem_classification.py    # T5Chem 分类 (API 不兼容, 未成功)
│
├── results/
│   ├── predictions/                    # 69 个 prediction CSV (23 models × 3 splits)
│   ├── tables/
│   │   ├── comparison_evans_temporal.csv
│   │   ├── comparison_evans_scaffold.csv
│   │   ├── comparison_evans_grouped_random_seed42.csv
│   │   ├── full_evans_*.json           # 详细指标 + 混淆矩阵 + CI
│   ├── models/                         # (未使用, 未保存 checkpoints)
│   └── figures/                        # (未生成)
│
├── external/                           # 外部 SOTA repos + 预训练权重
│   ├── drfp/                           # DRFP 反应指纹 (rule-based)
│   ├── rxnfp/                          # RXNFP (BERT-based reaction FP)
│   ├── bert-loves-chemistry/           # ChemBERTa pretrained weights
│   ├── Chemformer/                     # BART-based (Python 3.7, 未使用)
│   ├── ChiENN/                         # 手性感知 GNN
│   ├── EquiReact/                      # E(3)-equivariant 3D 反应网络
│   ├── GCPNet/                         # SE(3) GNN (蛋白质导向, 未适配)
│   ├── MolecularTransformer/           # Seq2seq (Python 3.5, 未使用)
│   ├── t5chem/                         # T5 化学模型 (API 不兼容)
│   └── pretrained_weights/
│       ├── DeepChem_ChemBERTa-77M-MLM/ # ChemBERTa (14MB, RoBERTa 3层)
│       └── laituan245_molt5-base/      # MolT5-base (945MB, T5 12层)
│
├── notebooks/
│   └── 01_data_cleaning_audit/         # SA 不一致性审查
│
├── logs/                               # 运行日志
│   ├── chienn.log
│   ├── equireact.log
│   └── conformer_gen.log
│
└── configs/                            # (空, 预留)
```

## Model Count by Type

| Type | Models | Count |
|------|--------|-------|
| Fingerprint + GBDT | XGBoost, LightGBM, XGBoost-FullFP, DRFP+XGB, DRFP+LGB, DRFP+Cond+XGB | 6 |
| Fingerprint + Other | RF, 1-NN, 5-NN, Morgan-MLP, RXNFP+XGB, RXNFP+LGB, RXNFP+MLP | 7 |
| Transformer | DistilBERT-Rxn, RoBERTa-Rxn, ChemBERTa-77M, MolT5-base | 4 |
| Reaction MPNN | Chemprop, Chemprop+Cond | 2 |
| Meta-learning | ProtoNet | 1 |
| Chemistry-informed DL | ChemAHNet-Aldol | 1 |
| 3D Models | ChiENN-Product, EquiReact | 2 |
| **Total** | | **23** |
