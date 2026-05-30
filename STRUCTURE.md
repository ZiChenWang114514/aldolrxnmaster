# Project Structure

```
aldolrxnmaster/
├── CLAUDE.md                  项目规范 (当前状态, 命名约定)
├── RESULTS.md                 V5 基准结果 (156d, 2427行, 9 辅基)
├── TODO.md                    路线图
│
├── chiralaldol/               核心包
│   ├── config.py              路径常量 + ML 常量
│   ├── data_io.py             数据加载 (prepare_Xy, load_splits, load_mechaware_bw)
│   ├── model_trainers.py      模型训练 (train_xgb, train_et, MajorityClassifier, ...)
│   ├── feature_registry.py    特征子集选择 (select_features, FEATURE_SUBSETS)
│   ├── rebuild/               V5 数据清洗管线 (13 步, 10 种辅基 SMARTS)
│   │   ├── constants.py       SMARTS, 溶剂 DB, 金属, 辅基催化排除
│   │   ├── audit.py           AuditTracker 类
│   │   └── step{01-12}_*.py   13 步管线模块 (含 step08b)
│   ├── rebuild_legacy/        V3 计算工具 (conformers, steric — 被 run_features.py 引用)
│   ├── steric_descriptors.py  %Vbur + Sterimol + 二面角 (24d)
│   ├── aldehyde_steric.py     醛 steric (10d)
│   ├── conformer_sampler.py   3D 构象采样
│   └── gnn/                   GNN 模块 (equiformer, schnet_3d, etc.)
│
├── scripts/                   可执行脚本 (15 个)
│   ├── run_rebuild.py         数据清洗 (134K Reaxys → 2434, 9 种辅基)
│   ├── run_features.py        特征工程 (→ 156d)
│   ├── run_splits.py          数据划分 (TSCV + scaffold + grouped)
│   ├── run_mechaware.py       MechAware 特征 (Z/E 分离 + BW 加权)
│   ├── run_benchmark.py       模型基准 (11 models × 10 splits)
│   ├── run_benchmark_full.py  完整基准 (含 MechAware)
│   ├── run_benchmark_evans.py Evans-only 基准
│   ├── run_optuna.py          Optuna 超参搜索
│   ├── run_optuna_benchmark.py Optuna 参数全 split 评估
│   ├── run_stacking.py        Stacking 集成
│   ├── run_chemprop.py        Chemprop MPNN baseline
│   ├── run_aux_models.py      Per-auxiliary 独立模型
│   ├── run_shap_analysis.py   SHAP 特征重要性
│   ├── run_error_analysis.py  错误分析
│   └── run_chem_space_audit.py 化学空间审计
│
├── data/
│   ├── clean_v5/              V5 清洗数据 (2434 行, 9 种辅基)
│   ├── features_v5/           156d 特征
│   ├── splits_v5/             TSCV + scaffold + grouped 划分
│   ├── clean_v4/              V4 历史数据 (保留)
│   └── features_v4/           V4 历史特征 (保留)
│
├── results/
│   ├── predictions_v5/        V5 预测 CSV (v4b, mechaware, steric, ablation, baseline)
│   ├── optuna/                Optuna 最优参数 + trials
│   ├── tables/                汇总表 (benchmark_v4.csv, ...)
│   ├── shap/                  SHAP 分析
│   └── chem_space_audit/      化学空间诊断
│
├── archive/                   归档 (V3 数据/脚本/notebooks, 废弃脚本)
│
├── literature/                文献资料
│
└── tests/                     单元测试
```
