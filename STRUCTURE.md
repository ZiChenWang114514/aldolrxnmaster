# Project Structure

```
aldolrxnmaster/
├── CLAUDE.md                  项目规范 (当前状态, 命名约定)
├── RESULTS.md                 V4 基准结果
├── MODEL_REGISTRY.md          模型注册表
├── LESSONS.md                 开发教训
├── TODO.md                    路线图
│
├── data/
│   ├── data.csv               原始 Reaxys 导出 (134,027 行, 41 列)
│   ├── clean_v4/              V4 清洗数据 (当前)
│   │   ├── substrate_aldol_clean.csv  2179 行, 5 种辅基, 38 列
│   │   ├── evans_clean.csv            1636 Evans 子集
│   │   ├── labels.csv                 4-class 标签
│   │   ├── condition_features.csv     44d 条件特征
│   │   └── audit/                     行级审计 + 步骤摘要
│   ├── features_v4/           V4 特征 (当前)
│   │   ├── v4_features.csv    2179 × 84d 完整特征矩阵
│   │   ├── steric_features.csv  34d steric
│   │   ├── labels.csv
│   │   ├── feature_manifest.json
│   │   └── conformers/        构象 pickle 缓存
│   ├── splits_v4/             V4 划分 (当前)
│   │   ├── tscv_fold{1-4}.json
│   │   ├── scaffold.json
│   │   └── grouped_seed{42,123,456,789,1024}.json
│   ├── raw/                   (已归档)
│   ├── clean/                 V3 清洗数据 (已被 V4 取代)
│   ├── features/              V3 特征 (已被 V4 取代)
│   └── splits/                V3 划分 (已被 V4 取代)
│
├── scripts/
│   ├── run_rebuild_v4.py      V4 12 步数据清洗 (134K → 2179)
│   ├── run_features_v4.py     V4 特征工程 (构象+steric+整合)
│   ├── run_splits_v4.py       V4 数据划分 (TSCV+scaffold+grouped)
│   ├── run_all_models_v4.py   V4 基准 (11 models × 10 splits)
│   ├── run_rebuild.py         V3 管线 (已取代)
│   ├── run_all_models_v3.py   V3 基准 (已取代)
│   └── ...                    其他旧脚本
│
├── chiralaldol/
│   ├── rebuild_v4/            V4 数据清洗管线 (12 步)
│   │   ├── constants.py       SMARTS, 溶剂 DB, 金属, 辅基催化排除
│   │   ├── audit.py           AuditTracker 类
│   │   ├── utils.py           SMILES 解析, 数值字段处理
│   │   └── step{01-12}_*.py   12 步管线模块
│   ├── rebuild/               V3 管线 (已取代)
│   ├── steric_descriptors.py  %Vbur + Sterimol + 二面角 (24d)
│   ├── aldehyde_steric.py     醛 steric (10d)
│   └── ...
│
├── results/
│   ├── predictions_v4/        V4 预测结果
│   │   ├── steric/            full_xgb, full_lgbm, full_et, full_rf, steronly_xgb
│   │   └── baseline/          cond_xgb, condaux_xgb, condaux_lgbm, knn_1, knn_5, majority
│   ├── tables/
│   │   └── benchmark_v4.csv   V4 基准汇总
│   └── predictions/           V3 预测结果 (已取代)
│
├── archive/                   归档
│   ├── data_raw_v3/           V3 原始 alldata.csv 备份
│   └── ...
│
└── tests/                     单元测试
```
