# Project Structure

```
aldolrxnmaster/
├── CLAUDE.md                  项目规范 (当前状态, 命名约定)
├── RESULTS.md                 V3 公平基准结果
├── MODEL_REGISTRY.md          47+ 模型注册表 (key, 状态, 指标)
├── LESSONS.md                 开发教训 (bug 修复经验)
├── TODO.md                    路线图
│
├── data/
│   ├── raw/                   原始数据 (alldata.csv, evans_aux*.csv)
│   ├── clean/                 清洗后数据
│   │   ├── evans_clean.csv    1655 Evans reactions (V3)
│   │   └── non_evans_clean.csv 430 non-Evans
│   ├── features/              特征文件
│   │   ├── v3_features.csv    87d 完整特征矩阵
│   │   ├── labels.csv         4-class 标签
│   │   ├── condition_features.csv  44d 条件特征
│   │   ├── steric_features.csv     24d 烯醇盐 steric
│   │   ├── aldehyde_steric.csv     10d 醛 steric
│   │   ├── feature_manifest.json   特征清单
│   │   └── mechaware/              MechAware Z/E 特征
│   │       ├── ketone_steric.csv   24d 酮式 steric
│   │       ├── z_enolate_steric.csv 24d Z-烯醇盐 steric
│   │       └── e_enolate_steric.csv 24d E-烯醇盐 steric
│   ├── splits/                数据划分 (JSON indices)
│   │   ├── tscv_fold{1-4}.json    4-fold temporal CV
│   │   ├── scaffold.json          Murcko scaffold split
│   │   └── grouped_seed*.json     5 grouped random splits
│   ├── interim/               中间产物 (调试用)
│   └── audit/                 行级审计报告
│
├── scripts/                   所有可执行脚本 (统一 run_*.py)
│   ├── run_rebuild.py         V3 16步数据重建管线
│   ├── run_mechaware_conformers.py  Z/E 构象生成
│   ├── run_mechaware.py       MechAware 模型训练
│   ├── run_all_models_v3.py   统一基准 (15 models × 10 splits)
│   ├── run_comparison.py      公平对比 + 泄漏检测
│   ├── run_chiralaldol.py     ChiralAldol 完整管线
│   └── run_tscv.py            时序交叉验证
│
├── chiralaldol/               ChiralAldol 核心方法
│   ├── enolate_generator.py   酮 → 烯醇盐转换
│   ├── ze_enolate_generator.py Z/E 烯醇盐 + 3D构象生成
│   ├── conformer_sampler.py   构象系综采样
│   ├── steric_descriptors.py  %Vbur + Sterimol + 二面角 (24d)
│   ├── aldehyde_steric.py     醛 steric (10d)
│   ├── feature_builder.py     V1-V5 特征组装
│   ├── solvent_lookup.py      溶剂推断 + KT 参数
│   ├── gnn/                   GNN 模块 (deprecated)
│   └── rebuild/               V3 16步重建管线模块
│       └── step00-16*.py
│
├── src/aldolrxnmaster/        核心包
│   ├── data/                  数据清洗 7步管线 (legacy)
│   ├── features/              特征计算
│   └── evaluation/            评估指标 (compute_all_metrics)
│
├── results/
│   ├── predictions/           模型预测 (按类别分目录)
│   │   ├── steric/            ChiralAldol + MechAware
│   │   ├── fp/                指纹模型 (含泄漏标注)
│   │   ├── gnn/               GNN (deprecated)
│   │   ├── meta/              ProtoNet
│   │   └── baseline/          基线模型
│   └── tables/                汇总表 CSV
│
├── archive/                   归档 (旧数据/旧预测/废弃脚本)
│   ├── data_processed/        旧 data/processed/ 全部内容
│   ├── predictions_legacy/    旧命名的 161 个 CSV
│   └── scripts_deprecated/    废弃脚本
│
├── external/                  外部依赖
│   ├── drfp/                  DRFP 指纹 (⚠️ 有泄漏)
│   └── rxnfp/                 RXNFP 指纹
│
└── notebooks/                 分析笔记本
    └── 02_shap_analysis/      SHAP + 误差分析
```
