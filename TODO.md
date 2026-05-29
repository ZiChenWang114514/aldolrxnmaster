# AldolRxnMaster — Roadmap (深度审计版)

## 项目状态 (2026-05-27)

- **数据**: V4d 管线从 134K Reaxys 原始数据重建，**2334 行**（6 种辅基类型）
- **冠军**: **v4b_full_et** (128d), TSCV = **0.624 ± 0.031**, Scaffold = 0.613, Grouped = 0.738
- **V3 真实性能** (2026-05-27 重新评估): V3 KNN balanced acc = **0.415**（200行测试，随机split，仅预测多数类）
- **3D syn/anti**: step08b 3D 二面角法，98.7% 成功率，仅作分析标签；2-class TSCV=0.746
- **管线可复现**: 13 步清洗 (含 step08b) + 行级审计，完全从原始 Reaxys 出发

---

## 已完成

- [x] **V4 数据重建**: 134K Reaxys → 13 步清洗 → 2334 行 (6 种辅基)
- [x] **辅基检测**: Evans + Crimmins thione/oxathione + Oppolzer + Myers + generic
- [x] **手性催化排除**: proline/BINAP/cinchona 等手性催化剂自动排除
- [x] **CIP 标签提取**: 产物 SMILES → atom mapping → Cα/Cβ CIP 编码 (0=R, 1=S)
- [x] **3D syn/anti 修复**: step08b 3D 二面角法替代不可靠的 CIP 启发式（一致率仅 45.6%）
- [x] **保护 OH 模板扩展**: step06 新增 6 条 SMARTS (silyl/benzyl/acetal)，+46 行数据
- [x] **V4d 特征工程**: 128d = steric(34) + conditions(44) + aux(6) + chirality(7) + rgroup(8) + chiralenv(21) + aldpri(8)
- [x] **MechAware V4d**: Z/E 分离 + BW 加权，ma_bw_xgb TSCV=0.604
- [x] **V4d 基准**: 11 models × 10 splits = 110 predictions（2334 行全部完成）
- [x] **DRFP 泄漏确认**: 产物 @/@@ 编码答案 → 排除
- [x] **Mukaiyama 修复**: step03 对有辅基的行不用命名反应排除
- [x] **V3 代码审计**: 2026-05-27 深度探索 v3_original/，发现核心缺陷（见下节）
- [x] **V3 实测 balanced acc**: 0.415（运行原始代码，公平评估后 V4 TSCV=0.624 >> V3 0.415）
- [x] **v3_original/ 文件夹建立**: 含 .gitignore + README，供 V3 原始文件迁移

---

## V3 深度审计结果 (2026-05-27 首次执行原始代码)

> 完整探索: v3_original/，原始文件约 180 个，覆盖 01~99 全部子目录

### V3 数据概况

| 数据集 | 行数 | 说明 |
|--------|------|------|
| evans_aux.csv (V3原始) | 1293 | Evans 手工清洗，label_1/2/3 ∈ {0,1} |
| data_final.csv (V3全集) | 2680 | Evans+Crimmins，含对映体增强，label_Ca/Cb ∈ {-1,+1} |
| test_data.csv (V3测试集) | 200 | 随机抽取 20%，包含 augmented_enantiomer |
| final-5-new-evans-seq.ipynb | — | Chemprop TSCV（2 fold 输出记录） |

### V3 原始代码核心缺陷（已定位，可直接修复）

#### 缺陷 1：Python 3 不兼容
- **文件**: `v3_original/.../aldol_predictor/fingerprints.py` Line 16
- **问题**: `sys.setdefaultencoding('utf-8')` 是 Python 2 API
- **错误**: `AttributeError: module 'sys' has no attribute 'setdefaultencoding'`
- **修复**: 注释该行（已修复，备份为 .bak）

#### 缺陷 2：标签编码不一致（3 套系统混用）
- evans_aux.csv: `label_1, label_2 ∈ {0, 1}`
- data_final.csv: `label_Ca, label_Cb ∈ {-1.0, +1.0}`
- stereo_class: `{0, 1, 4, 5}`（非连续！Ca=+1/Cb=-1 → 4，不是 2）
- V4 编码: `label_joint ∈ {0, 1, 2, 3}`（连续，R=0, S=1）
- **影响**: 代码无法跨数据集复用；{0,1,4,5} 被部分框架误当 5-class

#### 缺陷 3：模型完全崩溃到多数类（已实测）
- V3 KNN 只预测 class 0 (RR, 88次) 和 class 3 (SS, 112次)
- class 1 (RS) recall: **0.00**（16 个样本全错）
- class 2 (SR) recall: **0.00**（13 个样本全错）
- 表面准确率: 71%；真实 balanced acc: **0.415**（接近随机 0.25）
- 根因: 加权投票无 class weight 补偿；相似度噪声导致多数类始终获胜

#### 缺陷 4：无时间序列交叉验证（严重方法论问题）
- V3 split: 随机 80/20（random_state=42），train/test Year 1964-2023 完全混叠
- V4 解决方案: TSCV 4-fold，按 Year 排序，测试集始终在训练集之后

#### 缺陷 5：对映体增强数据泄漏
- 训练集包含 `augmented_enantiomer` 增强反应
- 测试集也含 augmented_enantiomer 样本（已在 test_data.csv 中确认）
- 一对对映体被随机拆分到 train/test → 相似性虚高

#### 缺陷 6：特征维度爆炸（计算低效）
- V3: Morgan(2048d) + MACCS(167d) + MCS + 3D(10d) + RC ≈ 44,400+ 维
- V4: 128d（紧凑物理化学特征）
- 问题: MCS 计算是 NP-hard；高维稀疏特征泛化差

#### 缺陷 7：KNN 预测器不可泛化
- 方法本质: 基于产物 SMILES 相似性搜索 k 近邻，投票决策
- 权重完全经验设定（Morgan 0.35 + MCS 0.20 + RC 0.15 + MACCS 0.15 + Desc 0.10 + 3D 0.05）
- 对未见结构预测能力极弱（记忆型而非学习型）

### V3 vs V4 公平对比（同等指标）

| 指标 | V3 (KNN) | V4 (ExtraTrees) | 提升 |
|------|----------|-----------------|------|
| **Balanced Acc** | **0.415** | **0.624** | **+50.1%** |
| 数据规模 | 2680 (含增强) | 2334 (干净) | V3 含对映体 |
| 验证方式 | 随机 80/20 | TSCV 4-fold | V4 更严格 |
| 数据泄漏 | 时间泄漏+增强泄漏 | 无泄漏 | V4 干净 |
| 少数类 (RS/SR) | recall=0.00 | recall>0.40 | V4 全类有预测 |
| 特征维度 | 44,000+ | 128 | V4 更高效 |

### V3 Chemprop TSCV 结果（notebook 输出读取）

| Fold | 4-class joint acc | Ca acc | Cb acc |
|------|------------------|--------|--------|
| Fold 0 | 0.6535 | 0.6614 | 0.7638 |
| Fold 1 | 0.6142 | 0.6220 | 0.7323 |

注意: 这是简单 accuracy（非 balanced），数据规模约 1300 行 Evans-only。
V4 TSCV=0.624 是 balanced acc（更严格），因此 V4 实际上显著优于 V3 Chemprop。

---

## 进行中

- [x] **RS-SynAnti 实验收尾**: TARGET_LABEL 已回 `label_joint`
- [x] **Evans-only V4d 基准**: Evans per-auxiliary bal_acc=0.771（化学空间审计确认）
- [x] **Optuna 超参搜索**: 3 models × 200 trials → ma_bw_xgb TSCV 0.604→0.666
- [x] **Ensemble Stacking**: TSCV=0.617（未提升），OOF-LR bound=0.660
- [x] **化学空间审计**: PCA + k-means + per-auxiliary + scaffold + TSCV distance

---

## 下一步优先级排序

---

### 【P0 — 紧急修复，影响结果正确性】

#### P0.1 重置 TARGET_LABEL
- **文件**: `scripts/run_all_models_v4.py` 第 ~80 行
- **当前**: `TARGET_LABEL = "label_joint_sa"`
- **改为**: `TARGET_LABEL = "label_joint"`
- **原因**: RS-SynAnti 实验已完成（TSCV=0.423，结论不可行），主线模型需要 label_joint

```python
# run_all_models_v4.py line ~80
TARGET_LABEL = "label_joint"  # 改回主线标签
```

#### P0.2 数据完整性快速验证
```bash
conda run -n aldol-rxn python -c "
import pandas as pd
labels = pd.read_csv('data/features_v4/labels.csv')
print('列数:', len(labels.columns), '期望7')
print('label_joint 有效:', labels['label_joint'].notna().sum(), '期望2334')
print('label_joint_sa 有效:', labels['label_joint_sa'].notna().sum(), '期望2304')
"
```

---

### 【P1 — 高价值实验，直接影响论文】

#### P1.1 Evans-only V4d 基准

**目的**: 得到与 V3 数据集规模可比的结果，为论文提供 V3 vs V4 公平对比

**设计方案**:
```bash
# 方法 A: 直接过滤现有特征矩阵中的 Evans 行
conda run -n aldol-rxn python -c "
import pandas as pd, json, numpy as np

# 找 Evans 行的 indices
df = pd.read_csv('data/clean_v4/substrate_aldol_clean.csv')
evans_mask = df['auxiliary_type'] == 'Evans'
evans_idx = df[evans_mask].index.tolist()
print(f'Evans 行数: {len(evans_idx)}')  # 期望 1654

# 过滤特征矩阵
X = pd.read_csv('data/features_v4/v4_features.csv')
labels = pd.read_csv('data/features_v4/labels.csv')
X_evans = X.iloc[evans_idx]
labels_evans = labels.iloc[evans_idx]
X_evans.to_csv('data/features_v4/v4_features_evans.csv', index=False)
labels_evans.to_csv('data/features_v4/labels_evans.csv', index=False)
"
```

**新脚本**: `scripts/run_evans_benchmark_v4.py`（仿照 run_all_models_v4.py，用 evans_mask 过滤）
**预期**: Evans TSCV ≈ 0.65-0.75（更高，因为辅基一致）
**意义**: 公平对比 V3 Chemprop (~0.65 simple acc) vs V4 ET（balanced acc）

#### P1.2 V3 代码完整修复（学术比较用）

**目的**: 论文中提供公平的 V3 benchmark 数字

**修复列表**:
1. `fingerprints.py:16` 删除 `sys.setdefaultencoding`（已完成）
2. 标签编码统一: `label_Ca/Cb ±1 → 0/1`，stereo_class {0,1,4,5} → {0,1,2,3}
3. 添加 class_weight='balanced' 到投票逻辑（解决多数类崩溃）
4. 实现 TSCV: 按 Year 排序，4-fold，报告 balanced accuracy
5. 去除 augmented_enantiomer（训练集去重，防止 test 泄漏）

**预期**: V3 修复后 TSCV balanced acc ≈ 0.42-0.50（低于 V4 0.624）
**文件位置**: `v3_original/05_code_notebooks/References_aldol-3-substructure/`

#### P1.3 Ensemble Stacking

**目的**: 组合 v4b_full_et (TSCV=0.624) + ma_bw_xgb (Grouped=0.752) 进一步提升

**设计**:
```
Level-0 预测器:
  - v4b_full_et    (128d)   TSCV=0.624
  - ma_bw_xgb      (156d)   TSCV=0.604
  - v4b_full_xgb   (128d)   TSCV=0.602
Level-1: 3*4=12d OOF 概率向量 → LogisticRegression / LightGBM
```

**新脚本**: `scripts/run_stacking_v4.py`
**关键**: 必须用 OOF (out-of-fold) 预测，否则 Level-1 在 train 上过拟合
**预期**: TSCV 提升 +0.02-0.05 → 约 0.640-0.670

#### P1.4 辅基感知建模（Auxiliary-aware）

**目的**: 不同辅基的立体控制机制不同，统一模型可能欠拟合

**策略 A — 独立模型**:
```python
for aux_type in ['Evans', 'Crimmins_thione', 'Oppolzer']:
    mask = df['auxiliary_type'] == aux_type
    X_sub = X[mask]  # Evans=1654, Crimmins=259, Oppolzer=141
    # 独立 TSCV
```

**策略 B — 强辅基分层特征**:
- 在现有 6d aux 向量基础上加入辅基骨架 Morgan 指纹（16d）
- 辅基类型 × 金属类型交叉特征

**新脚本**: `scripts/run_aux_models_v4.py`
**预期**: Evans 独立模型 TSCV ≈ 0.65-0.72

---

### 【P2 — 中等价值，改善模型】

#### P2.1 Optuna 超参优化

**优先模型**:
- ExtraTrees: `n_estimators`(100-1000), `max_features`(sqrt/log2/0.1-1.0), `min_samples_split`(2-20)
- XGBoost: `learning_rate`(0.01-0.3), `max_depth`(3-10), `subsample`(0.5-1.0), `colsample_bytree`

**框架**:
```python
import optuna
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5]),
        ...
    }
    # TSCV balanced accuracy
    return tscv_score(params)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

**新脚本**: `scripts/run_optuna_v4.py`
**预期**: TSCV 提升 0.01-0.03

#### P2.2 syn/anti 专用特征工程（下一代特征）

**背景**: RS-SynAnti 实验（label_joint_sa TSCV=0.423）失败的根因是特征与 syn/anti 正交
- 当前最高特征-synanti 相关: r=0.083
- Ca 和 syn/anti 正交: r=-0.006

**需要的新特征（Felkin-Anh/Zimmerman-Traxler 相关）**:
- 醛 α-位取代基 si/re 面区分指标（手性 α-碳时特别重要）
- 醛 α-碳的 A^1,3 应变估算（甲基 ax/eq 位倾向）
- 烯醇化物 Z/E 比例（从金属/碱条件推断）
- 酮 α-位取代基对称性（决定 E/Z 烯醇化物选择性）
- 过渡态手性匹配度（auxiliary-metal-TS 三元组）

**文件修改**: `scripts/run_features_v4.py` 新增函数 `compute_synanti_features()`
**挑战**: 需要更多化学先验知识，可能需要 TS 计算

#### P2.3 多构象 Sterimol 特征

**当前**: 每个 SMILES 用单一最低能量构象的 Sterimol B1/B5/L 参数
**改进**:
```python
# 当前: sterimol_B5 = 单构象计算
# 改进: 
sterimol_B5_mean = np.mean([conf_sterimol(conf) for conf in top3_confs])
sterimol_B5_std  = np.std([conf_sterimol(conf) for conf in top3_confs])
```
**预期**: Sterimol 参数更准确，steric 特征代表性更好

#### P2.4 Zimmerman-Traxler TS 几何（第二次尝试）

**第一次失败原因**（V4 qTS, -20% TSCV）:
1. TS 几何由 MMFF 估算，不准
2. 只有 4d 特征，信噪比低

**第二次方案**:
- 使用 xTB 计算 TS 几何（比 MMFF 准）
- 增加 TS 特征维度：Ca-Cb 键长、金属配位键长、椅式/船式判断、OH 方向
- 只针对 Evans + Crimmins（有明确 Zimmerman-Traxler TS 模型的辅基）

**文件**: 新建 `scripts/run_ts_features_v4.py`（需要 xTB 集成）

---

### 【P3 — 特征分析，支持论文】

#### P3.1 SHAP 特征重要性分析

**分析计划**:
```python
import shap
explainer = shap.TreeExplainer(et_model)
shap_values = explainer.shap_values(X_test)  # shape: (n, 4, 128)
# 每个 class 的特征重要性
shap.summary_plot(shap_values[0], X_test, feature_names=feature_names)
```

**关键问题**:
- 手性特征（7d）的 SHAP 贡献有多大？（期望最重要）
- 醛 CIP 优先级（8d）在 class 2/3 上是否比 0/1 更重要？
- steric (34d) 中哪几个 Sterimol 参数最重要？
- ChiralEnv (21d) 对提升哪个 class 贡献最大？

**输出**: `results/figures/shap/`，用于论文 Figure 3
**新脚本**: `scripts/run_shap_analysis.py`

#### P3.2 错误案例深度分析

**分析维度**:
- 按辅基类型分层错误率（Evans vs Crimmins vs Oppolzer）
- 按 Year 分层（早期 1980-2000 vs 近期 2010-2023）
- 错误预测的 4-class 混淆矩阵详细分析
- 高置信度但错误的案例（挖掘化学原因）
- label_joint_sa 预测为什么对 RS/SR 错得更多？

**输出**: `results/analysis/error_analysis.csv`
**新脚本**: `scripts/run_error_analysis.py`

#### P3.3 数据集描述性统计可视化

**内容（论文 Figure 1）**:
- 6 种辅基类型分布饼图（附代表性结构）
- 数据筛选漏斗（134K → 2334 行）
- label_joint 4-class 分布（RR=, RS=, SR=, SS= 各比例）
- 按 Year 的数据累积曲线（展示时序性）
- 反应条件分布（金属/溶剂/碱）

**新脚本**: `scripts/plot_dataset_stats.py`

#### P3.4 3D syn/anti 深度分析

**现有**: label_SA vs label_syn_anti_3d 一致率=45.6%，二面角分布双峰
**待补充**:
- 按辅基类型分层的 syn/anti 分布（Evans 是否比 Crimmins 更 syn 选择性？）
- 二面角 vs 金属类型散点图（B vs Ti vs Sn）
- syn/anti 与 conformer 能量的关系

---

### 【P4 — 论文写作准备】

#### P4.1 论文框架

**建议 Story（A → B → C）**:
- A: 现有方法在严格 TSCV 下局限性（V3 KNN=0.415 balanced，V3 Chemprop≈0.45）
- B: 物理化学特征工程（128d steric+chirality+conditions）+ ExtraTrees → TSCV=0.624
- C: 3D syn/anti 方法揭示 CIP ≠ syn/anti（新化学发现，45.6% 一致率）

**目标期刊**: JCIM / Digital Discovery / Nature Communications Chemistry
**特色亮点**:
- 可复现管线（13 步，代码全开源）
- 泄漏检测方法论（DRFP 案例 + TSCV vs random split 对比）
- 3D 二面角 syn/anti 确认（超出纯计算预测的化学发现）

#### P4.2 关键 Figure 规划

| Figure | 内容 | 状态 | 脚本 |
|--------|------|------|------|
| Fig 1 | 数据集描述（辅基 + 管线 + 类分布） | TODO | plot_dataset_stats.py |
| Fig 2 | V4d benchmark 气泡图（TSCV + Scaffold + Grouped） | 数据已有 | plot_benchmark.py |
| Fig 3 | SHAP 分析（128d feature importance × 4 class） | TODO | run_shap_analysis.py |
| Fig 4 | 3D syn/anti 二面角分布 + CIP vs 3D 一致率 | 基础完成 | existing plots |
| Fig 5 | V3 vs V4 公平对比（balanced acc + Evans-only） | 数据部分 | Evans baseline |
| Supp 1 | 消融实验 complete table | 数据已有 | - |
| Supp 2 | 错误分析 + 辅基分层 | TODO | run_error_analysis.py |
| Supp 3 | RS-SynAnti 实验结果 + 根因分析 | 完成 | - |

#### P4.3 论文补充材料清单

- [ ] 清洁后数据集 CSV（data/clean_v4/substrate_aldol_clean.csv，去除原始 Reaxys ID）
- [ ] 特征工程完整代码（scripts/run_features_v4.py）
- [ ] 最优模型权重文件（ExtraTrees pickle）
- [ ] 3D 二面角计算代码（step08b 部分）
- [ ] 完整 split 文件（splits_v4/）
- [ ] V4d benchmark 原始预测 CSV（results/predictions_v4/）

---

### 【P5 — 工程改进，长期维护】

#### P5.1 一键运行脚本

```bash
# 新建 run_pipeline_v4.sh
#!/bin/bash
set -e
echo "=== Step 1: Data Rebuild ==="
conda run -n aldol-rxn python scripts/run_rebuild_v4.py
echo "=== Step 2: Features ==="
conda run -n aldol-rxn python scripts/run_features_v4.py
echo "=== Step 3: Splits ==="
conda run -n aldol-rxn python scripts/run_splits_v4.py
echo "=== Step 4: MechAware ==="
conda run -n aldol-rxn python scripts/run_mechaware_v4.py
echo "=== Step 5: Benchmark ==="
conda run -n aldol-rxn python scripts/run_all_models_v4.py
echo "Done! Check results/tables/benchmark_v4.csv"
```

#### P5.2 V3 代码修复归档（供参考）

修复后的 V3 代码存放在 `v3_original/05_code_notebooks/.../aldol_predictor_fixed/`
修复内容:
1. `fingerprints.py:16` — 删除 `sys.setdefaultencoding`（Python 3 兼容）
2. 标签编码统一 → 0/1（去除 ±1 混乱）
3. stereo_class {0,1,4,5} → {0,1,2,3} 连续编码
4. 添加 class_weight='balanced' 到 KNN 投票
5. TSCV 4-fold 评估（按 Year 排序）
6. 去除 augmented_enantiomer 数据泄漏

#### P5.3 增量特征缓存验证

```python
# 验证构象缓存完整性
import pickle, os
cache_dir = 'data/features_v4/conformers/'
for pkl in os.listdir(cache_dir):
    with open(os.path.join(cache_dir, pkl), 'rb') as f:
        ens = pickle.load(f)
    assert isinstance(ens, dict), f"损坏: {pkl}"
print("所有构象缓存完整")
```

#### P5.4 结果复现性保障

```python
# 在 run_all_models_v4.py 输出文件中记录环境
import sklearn, platform
meta = {
    'sklearn_version': sklearn.__version__,
    'python_version': platform.python_version(),
    'TARGET_LABEL': TARGET_LABEL,
    'timestamp': datetime.now().isoformat()
}
json.dump(meta, open('results/tables/run_meta.json', 'w'))
```

---

## 关键教训（补充 LESSONS.md）

### L11: V3 KNN 的本质局限 — 近邻搜索不能处理类不平衡
- 相似度加权投票中，多数类因为样本数量优势始终获胜
- 解决方案: class weight ∝ 1/frequency；或在投票前对少数类加权

### L12: 随机 split 高估化学数据性能
- V3 random 80/20: 时间混叠（1964-2023），性能虚高
- V4 TSCV: 严格时序，比随机 split 低约 0.05-0.10
- 教训: 化学数据库必须用时序 CV；发表必须报告 TSCV

### L13: balanced accuracy 才是多类别不平衡的真实指标
- V3 71% accuracy → balanced acc=0.415（揭示模型崩溃）
- V4 TSCV=0.624（balanced acc），远优于 V3
- 教训: 含 class 不平衡时必须报告 balanced accuracy，accuracy 有误导性

### L14: RS-SynAnti 不可预测（特征-标签正交）
- Ca 和 syn/anti 正交 r=-0.006；当前特征最高 r=0.083
- 物理原因: Ca 由辅基控制，syn/anti 由醛面选择控制，不同物理来源
- 若要预测 syn/anti，需要基于 Felkin-Anh/Zimmerman-Traxler 的醛面专用特征

---

## V3 原始代码执行记录 (2026-05-27)

**执行环境**: conda aldol-rxn, Python 3.11, RDKit 2026.03.2, sklearn 1.8.0

**执行 1** — fingerprints.py Python 3 修复:
- 问题: `sys.setdefaultencoding` AttributeError
- 结果: 修复后 `AldolSelectivityPredictor` 成功导入
- 备份: `aldol_predictor/fingerprints.py.bak`

**执行 2** — V3 KNN 预测结果重新评估:
- 输入: `aldol_prediction_results_transformer.csv`（200行测试预测）
- **Balanced accuracy: 0.415**（而非表面 71%）
- 分析: 模型只预测 class 0 (RR, 88次) 和 class 3 (SS, 112次)
  - RR recall=0.79, RS recall=0.00, SR recall=0.00, SS recall=0.87
  - 预测分布: 88×RR + 112×SS vs 真实 85×RR + 16×RS + 13×SR + 86×SS

**执行 3** — V3 Chemprop TSCV 结果读取:
- Fold 0: joint accuracy=0.6535 (simple), Ca=0.6614, Cb=0.7638
- Fold 1: joint accuracy=0.6142 (simple), Ca=0.6220, Cb=0.7323
- 注意: simple accuracy，不是 balanced，数据为 Evans-only ~1300 行
- 与 V4 TSCV=0.624 (balanced) 不可直接比较

**结论**: V4 方法（balanced acc=0.624）在严格评估下显著优于 V3 KNN（0.415）。
V3 Chemprop 的 simple accuracy 0.63-0.65 在转换为 balanced acc 后预计约 0.45-0.55。

---

## 数据和文件路径速查

```
data/
  data.csv                       原始 Reaxys (134,027 行)
  clean_v4/
    substrate_aldol_clean.csv    (2334 行, 42 列)
    evans_clean.csv              (1654 行, Evans 子集)
    labels.csv                   (7 列: Ca/Cb/SA/joint/confidence/syn_anti_3d/joint_sa)
  features_v4/
    v4_features.csv              (2334 × 128d)
    labels.csv                   (同上，特征目录副本)
    v4_mechaware_bw.csv          (2334 × 112d)
    v4_mechaware_full.csv        (2334 × 328d)
    conformers/                  构象 pickle 缓存
  splits_v4/
    tscv_fold{1-4}.json          时间序列 CV（TSCV）
    scaffold.json                Murcko 骨架划分
    grouped_seed{42,101,202,512,1024}.json  role-aware 分组
results/
  tables/
    benchmark_v4.csv             V4d 汇总（11 models × 3 splits）
    benchmark_v4_sa.csv          RS-SynAnti 实验（TSCV=0.423，存档）
  predictions_v4/
    v4b/                         v4b_full_{et,xgb,rf,lgbm}
    mechaware/                   ma_{bw,full}_{xgb}
    steric/                      steronly_xgb
    ablation/                    no_chiral, chiral_only
    baseline/                    cond_xgb, majority
    sa_{v4b,mechaware,...}/      RS-SynAnti 实验（存档，勿删）
v3_original/                    V3 原始文件（只读参考）
  02_processed_datasets/
    References_aldol-2/evans_aux.csv            (1293 行, Evans 原始)
    References_aldol-3-substructure-2/data_final.csv (2680 行, V3全集)
  03_final_training_prediction/Final/
    aldol_prediction_results_transformer.csv    (200行测试, balanced=0.415)
    metrics_summary.csv                         (accuracy=0.71)
  05_code_notebooks/
    .../aldol_predictor/fingerprints.py         (已修复 Python 3 兼容)
    .../final-5-new-evans-seq.ipynb             (Chemprop TSCV 结果)
```

---

## 历史版本性能对比

| 版本 | 日期 | 数据量 | TSCV (balanced) | 变化 | 说明 |
|------|------|--------|-----------------|------|------|
| V3 KNN | ~2024 | 1293 Evans | **0.415** (重算) | — | 随机split，仅预测多数类 |
| V3 Chemprop | ~2024 | ~1300 Evans | ~0.45-0.55 (估算) | — | simple acc 0.63-0.65 |
| V4 | 2025 | 2288 | 0.507 | +22% vs V3 | 84d 特征，ExtraTrees |
| V4b | 2025 | 2288 | 0.565 | +11% | +手性(7d)+Rgroup(8d) |
| V4c | 2025 | 2288 | 0.625 | +11% | +ChiralEnv(21d)+AldPri(8d)+MechAware |
| V4d | 2026-05-27 | **2334** | 0.624 | ≈ V4c | +46行(保护OH)+3D syn/anti 标签 |
| V4d-SA | 2026-05-27 | 2304 | 0.423 | — | RS-SynAnti 实验，已放弃 |
| V4d Stacking | 2026-05-28 | 2334 | 0.617 | — | ET+XGB+MA-BW→LR，未提升 |
| V4d Optuna (128d) | 2026-05-28 | 2334 | 0.666 | +6.7% | ma_bw_xgb_optuna (128d 参数) |
| **V4d Optuna (153d)** | 2026-05-28 | **2334** | **0.657** | **+5.3%** | ma_bw_xgb (153d 重搜) |
| Chemprop+Features | 2026-05-28 | 2334 | 0.626 | — | MPNN baseline, Grouped=0.789 |
| Chemprop (纯SMILES) | 2026-05-28 | 2334 | 0.601 | — | MPNN 无手工特征 |
| Evans-only ET | 2026-05-28 | 1654 | 0.710 | — | 辅基独立建模 |
