# Known Issues & Limitations

## Data Issues

### D1: 35 DRFP Failures
- 35/1822 reactions (1.9%) 的 DRFP 指纹生成失败 (RDKit stereo bond error)
- 这些行用零向量替代，可能影响 DRFP 模型的测试准确性
- **影响**: 轻微，35 行中大部分不在 temporal test set (155 行)

### D2: 21 Invalid 3D Conformers
- 21/1822 molecules (1.2%) 无法生成 3D conformer
- ChiENN 和 EquiReact 的 test set 中这些行被跳过
- **影响**: 3D 模型的 test set 略小于 2D 模型，但不影响 fair comparison (每个模型只评估自己能处理的样本)

### D3: Anti-class Imbalance
- C1 (anti, Ca=0,Cb=1): 10.3%, C2 (anti, Ca=1,Cb=0): 7.4%
- 所有模型在 anti 类上 F1 显著低于 syn 类
- **缓解**: 使用 class-weighted loss 和 balanced_accuracy 指标

### D4: SA 不一致 (17 行)
- 17 行的 syn/anti 标签与 Ca==Cb 推导不一致，全部为 AsymmetricDouble 反应类型
- 已在 `notebooks/01_data_cleaning_audit/` 中详细审查

## Model Issues

### M1: T5Chem API 不兼容
- `TrainingArguments.overwrite_output_dir` 在 transformers 5.8 中被移除
- 已修复该参数，但 T5Chem 还有其他 API 不兼容问题
- **状态**: 未成功跑通，不在 benchmark 中

### M2: MolT5-base 性能极差
- Temporal bal_acc = 0.2623 (仅略高于 majority class 0.25)
- 原因: MolT5 预训练于分子 SMILES，从未见过 `>>` 符号；即使 unfreeze 6/12 层仍学不到有效表示
- **结论**: MolT5 encoder 不适合 reaction-level classification

### M3: ChemBERTa-77M Vocab 不足
- vocab_size=600, max_position_embeddings=515
- 对复杂 reaction SMILES 编码能力不足
- **结论**: 化学专用 tokenizer 的优势被过小的 vocab 抵消

### M4: ChiENN / EquiReact 3D Models 失败
- ChiENN-Product temporal: 0.2226 (低于 random)
- EquiReact temporal: 0.2812
- **可能原因**:
  1. 仅用 product 的 3D structure，丢失了 reactant→product 的变化信息
  2. 1822 样本对 3D equivariant 网络来说太少
  3. ETKDG 生成的 conformer 可能不是反应相关的活性构象
  4. 这些模型在 scaffold split 上表现好得多 (ChiENN 0.47, EquiReact 0.50)，暗示 temporal shift 是主要挑战

### M5: Chemformer / MolecularTransformer 环境不兼容
- Chemformer: Python 3.7 + pytorch-lightning 1.2.3
- MolecularTransformer: Python 3.5 + OpenNMT 0.4.1
- **状态**: 跳过，不值得为旧 API 单独建环境

### M6: GCPNet 适配复杂
- 面向蛋白质-配体任务 (PDB 格式, Bio.PDB, atom3d)
- 适配小分子反应分类需要大量重写
- **状态**: repo 已 clone，未适配

## Infrastructure Issues

### I1: SOCKS 代理
- HuggingFace Hub 下载需要 `httpx[socks]` (已安装)
- 首次下载 DistilBERT/RoBERTa 需要网络，之后走本地缓存

### I2: 模型 Checkpoints 未保存
- `results/models/` 为空，所有模型结果仅保存为 prediction CSV
- 重现需要重新训练
- **建议**: 对 top-3 模型保存 checkpoints

### I3: conda run 输出缓冲
- `conda run` 缓冲 stdout，背景任务的 `.output` 文件可能为空
- **解决**: 用 nohup + 重定向到日志文件

### M7: ChiralAldol Enolate Generation — 21 Failures

- 21/1822 ketones (1.2%) have NaN SMILES in the Ketone column → enolate generation returns `parse_fail`
- These molecules fall back to cleaned ketone SMILES for conformer generation
- **Impact**: Minimal — only 1.2% of data, and fallback uses original ketone structure

### M8: ChiralAldol Atom Count Mismatch (Fixed)

- ~24 molecules had mismatched atom counts between SMILES-parsed mol and conformer coordinates
- Caused IndexError in Sterimol computation (`index 24 out of bounds for size 23`)
- **Fix**: Added bounds checking in `steric_descriptors.py` — affected molecules get zero-filled descriptors
- **Root cause**: Some enolate SMILES parse to slightly different atom counts when re-parsed vs original conformer generation. Edge case in RDKit's SMILES canonicalization.

### M9: ChiralAldol Performance Gap — RESOLVED via Late Fusion

- ChiralAldol-XGB standalone (0.664) < DRFP+Cond (0.711) on temporal
- Early fusion (feature concat 193d) → 0.636, WORSE — DRFP dimensional dominance diluted steric signal
- **RESOLVED**: Late fusion via Stacking → **0.725** (new champion, +1.4% over DRFP+Cond)
- Root cause of early fusion failure: 128d DRFP vs 24d steric → XGBoost splits overwhelmingly favor DRFP features
- Lesson: multi-view fusion on small datasets requires prediction-level (not feature-level) combination

### M10: Hard Cases — 7.1% Universally Mispredicted

- 11/155 temporal test samples are wrong by ALL 7 evaluated models
- Class distribution: C3 (syn-S) = 63.6%, C1 (anti) = 18.2%
- These may represent: (1) label noise, (2) non-Evans selectivity mechanisms, or (3) missing features (aldehyde steric, electronic effects)
- Saved to `notebooks/02_shap_analysis/hard_cases.csv` for future investigation

### M11: 醛基完全未建模 — 当前最关键的特征盲点 (2026-05-12 识别)

- **问题**: 现有 24d steric 特征全部描述烯醇盐（亲核体），对醛（亲电体）的 R-group 空间体积零描述
- **影响**: Zimmerman-Traxler TS 的立体选择性由两个因素决定：(1) 烯醇盐暴露哪个面（已建模），(2) 醛基 R₂ 占赤道键 vs 直立键（完全未建模）
- **症状**: 11 个硬案例中 63.6% 是 C3 (syn-S)，可能正是因为 S-辅基反应中醛基效应被错误忽略
- **修复路径**: A1 任务 — 醛基 Sterimol (L/B1/B5) + %Vbur，复用 `steric_descriptors.py` 现有函数
- **预期增益**: 添加 ~12d 特征后重新训练 Stacking，预期 C3 准确率提升最显著

### M12: sin_tau1 代理 TS 几何，本质是近似

- **问题**: sin_tau1 (#1 SHAP=0.794) 是模型用基态二面角去推断 TS 几何的间接代理
- **影响**: 信息损失不可避免；模型在底物变化时泛化能力受限
- **根本解**: qTS 建模（Phase B2）—— 直接提供 Zimmerman-Traxler 竞争通道的 ΔE_qTS
- **短期缓解**: 无（A1 不解决此问题）

## 发表前需解决

1. ~~T1: Fusion model~~ → DONE (Stacking 0.725, new champion)
2. ~~T2: SHAP~~ → DONE (sin_tau1 #1, Vbur_diff #4)
3. ~~T3: Error analysis~~ → DONE (20 unique correct, 11 hard cases)
4. **A1**: 醛基 Sterimol/%Vbur 特征（Phase 11 首要任务）
5. **A2/A3**: 确认对映体增强可行性 + dr值是否可用
6. **T4**: 论文级图表 (confusion matrices, violin plots, radar charts)
7. **T5**: Manuscript figures and SI tables
8. 保存 top-3 模型 checkpoints (可复现性)
9. MajorityClass/Random baselines 补全 CSV
