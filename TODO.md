# AldolRxnMaster — Roadmap (深度审计版)

## 项目状态 (2026-06-05)

- **数据**: V5 管线从 134K Reaxys 原始数据重建，**2434 行**（9 种辅基类型 + 7 other）
- **V5 辅基**: Evans (1661) + Crimmins thione (260) + Crimmins oxathione (169) + Oppolzer (141) + **Abiko (127)** + **Menthyl ester (32)** + **Oxazoline (21)** + **Myers (16)** + Other (7)
- **VALID**: 10 种辅基, **2427 行**, **156d** 特征
- **冠军 (Evans-only)**: **ZT-Chiral+feat** (ZT 图 + 156d), TSCV = **0.818 ± 0.017**
- **冠军 (全数据集)**: **XGB Optuna** (156d), TSCV = **0.739 ± 0.074**, Grouped = **0.760**
- **V3 真实性能** (2026-05-27 重新评估): V3 KNN balanced acc = **0.415**（200行测试，随机split，仅预测多数类）
- **3D syn/anti**: step08b 3D 二面角法，97.9% 成功率，仅作分析标签
- **管线可复现**: 13 步清洗 (含 step08b) + 行级审计，完全从原始 Reaxys 出发
- **代码清理**: 2026-06-05 完成大规模重构，41→22 脚本，消除 ~1700 行重复代码

---

## 已完成

- [x] **V5 数据重建**: 134K Reaxys → 13 步清洗 → 2434 行 (9 种辅基 + 7 other)
- [x] **V5 辅基扩展**: +Abiko(127)/Menthyl(32)/Oxazoline(21)/Myers 放宽(16)/Ynamide 排除(47)
- [x] **V5 特征工程**: 156d = Steric(34) + Conditions(44) + Aux(15) + Chirality(7) + R-group(7) + ChiralEnv(21) + AldPri(8) + DeltaChiral(16) + ChiralDet(3) + n_stereo(1)
- [x] **V5 Benchmark**: 默认 ET TSCV=0.677, Optuna XGB TSCV=0.739, Scaffold=0.831
- [x] **ZT-GNN Evans Benchmark**: 7 模型, ZT-Chiral+feat TSCV=0.818 (Evans-only 冠军)
- [x] **SPMS 球面投影特征**: 16d stats + 24d face map, XGB+face_map TSCV≈0.685
- [x] **Chemprop MPNN**: SMILES-only TSCV=0.601, +156d=0.626, +ZT=0.809
- [x] **代码清理**: sys.path 去除, dead code 清除, DRY 重构 (2026-06-05)
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

## 已完成的实验 (V5 + ZT-GNN)

- [x] **V5 Optuna 超参搜索**: XGB TSCV=0.739, ET=0.722, MA-BW-XGB=0.666
- [x] **ZT 过渡态图构建**: Evans 99.4% 覆盖率 (1644/1654), 模板拼接法
- [x] **ZT-GNN 7 模型 benchmark**: Chiral=0.818, ComENet=0.784, Hybrid=0.776, GAT=0.753, GIN=0.731
- [x] **SPMS 全流程**: Phase A/B/C + 13 benchmark 实验
- [x] **SHAP 分析**: 153d × 4-class, top=ald_pri_priority_proxy
- [x] **错误分析**: RR→SS 最常见 (CIP 翻转), 0 高置信度错误, 259 标签候选
- [x] **化学空间审计**: distance-accuracy r=-0.916, 5 clusters 对应辅基类型

## 负面结果 (已排除)

- **MultiTS 多 TS 注意力** (2026-06-05): fold1-2 ~0.715 vs ZT-Chiral 0.818, 放弃 (stash 保存)
- **xTB 电子特征** (12d): -35% TSCV, 噪声过大
- **qTS 过渡态特征** (4d): -20% TSCV, MMFF TS 不准确
- **DRFP 指纹**: 产物 @/@@ 直接编码答案 → 标签泄漏
- **Stacking 集成**: TSCV=0.617, 未提升 (inner-val 数据不足)
- **RS-SynAnti 预测**: TSCV=0.423, 特征与 syn/anti 正交 (r=0.083)

---

## 下一步优先级排序 (2026-06-05 更新)

---

### 【P0 — 全部已完成】 ✅

- [x] TARGET_LABEL 已回 `label_joint`
- [x] V5 数据完整性验证通过
- [x] V4→V5 路径迁移完成 (config.py)

---

### 【P1 — 数据质量 & 模型扩展】

#### P1.1 新辅基标签质量审查 (高优先级, 低成本)
- **Menthyl ester**: balanced_acc=0.250 (完全随机), 32 行 — 是否标签有系统性错误?
- **Oxazoline**: balanced_acc=0.500, 21 行 — 样本量还是标签噪声?
- **方法**: 手动抽查 CIP 标签 + 原始 Reaxys 反应条件

#### P1.2 ZT 图扩展到 Crimmins/Oppolzer (中优先级)
- Crimmins thione/oxathione 也遵循 Zimmerman-Traxler TS 模型 (Ti 金属, 6-membered chair)
- Oppolzer 用 B(OBu₂) enolate，也有类似 TS
- 预期: Evans+Crimmins+Oppolzer = 2231 行 (92% 数据) 可用 ZT-GNN
- 如成功，全数据集 GNN 冠军可能显著提升

#### P1.3 混合模型: ZT-GNN + Tree fallback (中优先级)
- 对有 ZT 图的辅基 (Evans/Crimmins/Oppolzer) 用 GNN
- 对 Abiko/Menthyl 等无 ZT 的辅基用 Tree
- 论文叙事: "mechanistically informed model selection"

---

### 【P2 — 论文写作准备】

#### P2.1 论文框架

**Story (A → B → C)**:
- A: 手性辅基 aldol 立体选择性预测, 现有方法局限 (V3 KNN balanced=0.415)
- B: 156d 物理化学特征 + ZT 过渡态图 (ZT-GNN), 严格 TSCV 评估
- C: Evans ZT-Chiral TSCV=0.818, 全数据集 XGB=0.739; 3D syn/anti 揭示 CIP≠syn/anti (45.6%)

**目标期刊**: JCIM (首选) / Digital Discovery / JACS (如 ZT 扩展成功)

#### P2.2 Figure 制作

| Figure | 内容 | 数据状态 |
|--------|------|----------|
| Fig 1 | 数据集描述: 辅基分布 + 筛选漏斗 + 4-class 分布 | 数据已有, 需制图 |
| Fig 2 | 模型 benchmark 气泡图 (Tree vs GNN vs MPNN) | 数据已有, 需制图 |
| Fig 3 | ZT 过渡态图示意 + ZT-GNN vs Tree 对比 | 数据已有, 需制图 |
| Fig 4 | SHAP 特征重要性 (top-20, 按 class 分层) | SHAP 已有, 需制图 |
| Fig 5 | 3D syn/anti 二面角分布 + CIP vs 3D | 基础完成 |
| Supp | 消融实验 + Per-aux 热图 + 错误分析 + 负面结果 | 数据已有 |

#### P2.3 论文补充材料
- [ ] 清洁后数据集 CSV（去除原始 Reaxys ID）
- [ ] 特征工程完整代码
- [ ] 最优模型权重文件
- [ ] 3D 二面角计算代码
- [ ] 完整 split 文件 + benchmark 预测 CSV

---

### 【P3 — 低优先级实验, 可作 future work】

#### P3.1 Felkin-Anh/Zimmerman-Traxler syn/anti 特征
- 当前特征与 syn/anti 正交 (r=0.083)
- 需要全新的醛面选择性特征 (si/re 面区分, A^1,3 应变, Z/E 烯醇化物比例)
- 可在论文 Discussion 中提出

#### P3.2 Multi-conformer Sterimol
- 单构象 → top-3 构象 Sterimol 平均/标准差
- 预期提升小 (+0.01-0.02)

#### P3.3 V3 代码修复 (学术比较用)
- 已有 V3 KNN balanced=0.415 的数字
- 完整修复 V3 代码的投入产出比低

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

## 数据和文件路径速查 (V5)

```
data/
  data.csv                       原始 Reaxys (134,027 行)
  clean_v5/
    substrate_aldol_clean.csv    (2434 行, 42 列, 9 种辅基)
    evans_clean.csv              (1661 行, Evans 子集)
    labels.csv                   (标签: Ca, Cb, SA, joint + 3D syn/anti)
    condition_features.csv       (44d 条件特征)
    audit/                       行级审计报告
  features_v5/
    v5_features.csv              (2427 × 156d 完整特征矩阵)
    v5_features_spms.csv         (2427 × 172d SPMS 增强特征)
    steric_features.csv          (34d 空间位阻特征)
    mechaware_bw.csv             (BW 加权 MechAware 特征)
    mechaware_full.csv           (完整 MechAware 特征)
    labels.csv
    conformers/                  构象 pickle 缓存
    spms/                        SPMS 矩阵 + face map
    zt_graphs/                   evans_zt_graphs.pkl
  splits_v5/
    tscv_fold{1-4}.json          时间序列 CV（TSCV）
    scaffold.json                Murcko 骨架划分
    grouped_seed{42..1024}.json  role-aware 分组划分
  clean_v4/, features_v4/        V4 历史数据 (保留，不再修改)
results/
  predictions_v5/
    v4b/                         XGB/ET/RF/LGBM (156d)
    mechaware/                   MechAware-BW/Full-XGB
    steric/                      位阻-only
    ablation/                    特征消融
    baseline/                    多数类/条件
    chemprop/                    Chemprop MPNN
    zt_chemprop/                 ZT+Chemprop
    zt_gnn/                      ZT-GIN/GAT/Chiral (+feat)
    optuna/                      Optuna-tuned
    spms/                        SPMS Tree 模型
  optuna/                        Optuna 最优参数 JSON
  tables/                        汇总表
  shap/                          SHAP 分析
v3_original/                    V3 原始文件（只读参考）
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
| V4d | 2026-05-27 | 2334 | 0.624 | ≈ V4c | +46行(保护OH)+3D syn/anti 标签 |
| V4d Optuna (153d) | 2026-05-28 | 2334 | 0.657 | +5.3% | ma_bw_xgb (153d 重搜) |
| **V5 default ET** | 2026-05-30 | **2427** | **0.677** | — | 10 种辅基, 156d |
| **V5 Optuna XGB** | 2026-05-30 | **2427** | **0.739** | **+9.2%** | 200 trials, 全数据集冠军 |
| V5 Chemprop+156d | 2026-05-30 | 2427 | 0.626 | — | MPNN baseline |
| **V5 ZT-Chiral** | 2026-05-30 | **1661 Evans** | **0.818** | **+15.2%** | Evans-only GNN 冠军 |
| V5 ZT-Chemprop+ZT | 2026-05-30 | 1661 Evans | 0.809 | — | MPNN + ZT 32d |
| V5 MultiTS | 2026-06-05 | 1661 Evans | ~0.715 | — | 负面结果，已放弃 |
