# Project Structure

```
aldolrxnmaster/
├── CLAUDE.md                          # Claude Code 项目指引
├── RESULTS.md                         # 45+ 模型 benchmark 结果
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
│   │   ├── evans_clean.csv            # 原始 Evans 清洗数据 (1822 行, legacy)
│   │   ├── evans_v2_clean.csv         # Phase A 清洗后 (1801 行, 当前使用)
│   │   ├── all_clean.csv              # 全量清洗数据 (4258 行, Evans+非Evans)
│   │   ├── non_evans_clean.csv        # 非 Evans 子集 (2457 行)
│   │   ├── quality_audit.csv          # 行级质量审计 (confidence 标签)
│   │   ├── features/
│   │   │   ├── labels.csv             # 标签 (1801 行, label_joint etc.)
│   │   │   ├── reaction_smiles.csv    # 清洗后反应 SMILES
│   │   │   ├── reaction_conditions.csv # 反应条件 (35d, 含溶剂推断)
│   │   │   ├── auxchiral_features.csv # 辅基手性 (6d)
│   │   │   ├── drfp_fps.npz           # DRFP (1801×2048)
│   │   │   └── rxnfp_fps.npz          # RXNFP (1801×256)
│   │   ├── splits/
│   │   │   ├── evans_temporal.json     # 时间分割 (1801 行重建)
│   │   │   ├── evans_scaffold.json     # Murcko scaffold 分割
│   │   │   └── evans_grouped_random_seed*.json  # 分组随机 (5 seeds)
│   │   ├── graphs/                     # GNN 图表示 (Phase B)
│   │   │   ├── diff_graphs.pt          # 反应差异图 (1801, atom-mapped)
│   │   │   ├── multiview_graphs.pt     # 多视图图 (reactant+product)
│   │   │   ├── spatial_3d_graphs.pt    # 3D 空间图 (conformer coords)
│   │   │   ├── ts_approx_graphs.pt     # TS 近似图 (bond change)
│   │   │   └── all_diff_graphs.pt      # 全量差异图 (4258, 迁移学习用)
│   │   └── conformers/
│   │       └── conformers.pkl          # 3D 构象 (ETKDG+MMFF)
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
│   ├── run_chiralaldol_pipeline.py     # ChiralAldol 全管线 (enolate→conf→steric→train)
│   ├── run_data_audit.py               # Phase A: 数据审计+清洗 (1822→1801)
│   ├── run_prepare_all_data.py         # Phase B2: 全量数据集 all_clean.csv
│   ├── run_timeseries_cv.py            # Phase A4: Time-series CV (4-fold temporal)
│   ├── run_build_graphs.py             # Phase B4: 4 种 GNN 图表示构建
│   ├── run_gnn_benchmark.py            # Phase C: GNN 12 组合 coarse screening
│   ├── run_transfer_learning.py        # Phase D: 迁移学习 (部分完成)
│   ├── rebuild_comparison.py           # 从 prediction CSVs 重建统一对比表
│   └── run_v5_pipeline.py              # V5 交叉项 (负面结果)
│
├── chiralaldol/                           # ChiralAldol 核心模块
│   ├── enolate_generator.py               # M1: 酮 → Z/E 烯醇盐
│   ├── conformer_sampler.py               # M2: 构象系综采样 (100 conf → RMSD 聚类)
│   ├── steric_descriptors.py              # M3: 烯醇盐 3D 立体 (%Vbur, Sterimol, 二面角, 24d)
│   ├── aldehyde_steric.py                 # M3b: 醛基 3D 立体 (Sterimol, Vbur, 10d)
│   ├── solvent_lookup.py                  # 溶剂推断 + Kamlet-Taft 参数表
│   ├── xtb_descriptors.py                 # B1: GFN2-xTB 电子 12d (负面结果)
│   ├── qts_builder.py                     # C1: qTS VDW 立体 4d (负面结果)
│   ├── feature_builder.py                 # M4: 特征集成 (V1/V2/V3/V3b/V5)
│   ├── utils.py                           # 工具函数
│   └── gnn/                               # GNN 模块 (Phase C)
│       ├── graph_builder.py               # 4 种图表示构建
│       ├── mpnn_diff.py                   # G1: MPNN on 反应差异图
│       ├── gat_multiview.py               # G2: GAT 多视图 (reactant+product)
│       ├── equiformer.py                  # G3: SE(3)-equivariant Transformer
│       ├── schnet_3d.py                   # G4: SchNet 3D 连续卷积
│       ├── condition_fusion.py            # FiLM / Concat / Inject 融合
│       └── trainer.py                     # 统一训练循环
│
├── results/
│   ├── predictions/                    # prediction CSVs (47+ tabular + GNN)
│   ├── tables/
│   │   ├── comparison_evans_temporal.csv   # 全模型对比
│   │   ├── gnn_coarse_screening.csv        # GNN 12 组合结果
│   │   ├── tscv_results.json               # Time-series CV 4-fold
│   │   └── full_evans_*.json               # 详细指标 + CI
│   ├── models/                         # (未保存 checkpoints)
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
| **ChiralAldol** | V1-XGB, V2-XGB, V2-Stack, V3/V3b/V4/V5-XGB, V5-LGBM/ET/Stack/V5s, SterOnly, CondAux, WtVote, Stack, +DRFP | 17 |
| AuxChiral | XGB, +Ald-XGB, LGBM, NoAux, NoBase | 5 |
| DRFP fusion | +XGB, +LGB, +Cond+XGB, +Aux+Cond | 4 |
| FP + GBDT | XGBoost, LightGBM, RF, XGBoost-FullFP | 4 |
| FP + Other | 1-NN, 5-NN, Morgan-MLP, RXNFP+XGB/LGB/MLP | 6 |
| Transformer | DistilBERT-Rxn, RoBERTa-Rxn, ChemBERTa-77M, MolT5-base | 4 |
| Reaction MPNN | Chemprop, Chemprop+Cond | 2 |
| Meta-learning | ProtoNet | 1 |
| Chemistry-informed DL | ChemAHNet-Aldol | 1 |
| 3D Models | ChiENN-Product, EquiReact | 2 |
| Baselines | MajorityClass, Random | 2 |
| **Total (unique predictions)** | | **47** |
