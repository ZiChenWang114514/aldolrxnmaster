# Results — AldolRxnMaster

Evans 不对称 Aldol 反应 4-class 立体化学预测基准。

## V3 公平基准 (2026-05-16)

**数据**: 1655 Evans reactions (V3 严格清洗)  
**划分**: TSCV 4-fold temporal + Scaffold (Murcko) + Grouped random (5 seeds)  
**评估**: balanced accuracy (macro-averaged recall, 4-class)  
**无泄漏**: 角色保留 group_id，所有划分经过泄漏验证

### Champion: MechAware-Full (TSCV = 0.733)

### 完整排名

| Rank | Key | Full Name | Category | Dim | TSCV mean±std | Scaffold | Grouped mean |
|------|-----|-----------|----------|-----|---------------|----------|--------------|
| 1 | ma_full | **MechAware-Full** | steric | 151d | **0.733±0.074** | 0.672 | 0.732 |
| 2 | ma_bw | MechAware-BW | steric | 79d | 0.725±0.060 | 0.652 | 0.723 |
| 3 | cv2_xgb | ChiralAldol-V2-XGB | steric | 78d | 0.696±0.023 | 0.737 | 0.731 |
| 4 | condaux_xgb | CondAux-XGB | baseline | 53d | 0.689±0.086 | 0.666 | 0.680 |
| 5 | cv2_rf | ChiralAldol-V2-RF | steric | 78d | 0.680±0.049 | 0.706 | 0.726 |
| 6 | cv2_et | ChiralAldol-V2-ET | steric | 78d | 0.665±0.039 | 0.709 | 0.741 |
| 7 | cv2_lgbm | ChiralAldol-V2-LGBM | steric | 78d | 0.649±0.052 | 0.726 | 0.720 |
| 8 | knn_1 | 1-NN | baseline | 78d | 0.599±0.090 | 0.679 | 0.646 |
| 9 | knn_5 | 5-NN | baseline | 78d | 0.576±0.066 | 0.646 | 0.584 |
| 10 | steronly_xgb | StericOnly-XGB | steric | 24d | 0.514±0.053 | 0.535 | 0.587 |
| 11 | cond_xgb | CondOnly-XGB | baseline | 44d | 0.392±0.028 | 0.394 | 0.440 |
| 12 | majority | MajorityClass | baseline | — | 0.250±0.000 | 0.250 | 0.250 |

### ⚠️ 泄漏模型（已排除）

| Key | Full Name | Reported TSCV | Leakage Mechanism |
|-----|-----------|---------------|-------------------|
| drfp_xgb | DRFP-XGB | ~~0.849~~ | 产物 SMILES @/@@ → DRFP shingles 直接编码立体化学 |
| drfp_cond_xgb | DRFP+Cond-XGB | ~~0.872~~ | 同上 |

---

## 泄漏检测 (2026-05-16)

### 1. DRFP 标签泄漏

**机制**: DRFP (Differential Reaction Fingerprint) 对 `reactants >> product` SMILES 做 shingle 差集。产物 SMILES 中的 `@`/`@@` 标记直接编码了立体化学构型（即我们预测的标签）。DRFP 的 shingles 保留了这些标记，模型不需要学化学就能"看到答案"。

**证据**: DRFP TSCV=0.87 远超其他模型（次好 0.73），且 DRFP fingerprint 对同一反应不同立体异构体产生完全不同的位模式。

### 2. 旧划分 group_id 泄漏

**机制**: 旧 `deduplicate.py` 用 `sorted([ketone, aldehyde])` 字母排序做去重 key，破坏了 ketone/aldehyde 角色区分。

**证据**:
| Split | 旧 group_id | 新 group_id (修复) | Delta |
|-------|-------------|-------------------|-------|
| Scaffold | 0.826 | 0.757 | **-6.9%** ⚠️ |
| Grouped | 0.807 | 0.708 | **-9.8%** ⚠️ |
| Temporal | 0.673 | 0.758 | +8.5% (不受影响) |

---

## Key Findings

1. **手工 3D 特征仍是最佳方法** — MechAware (151d) 包含 Z/E 烯醇盐分离 steric 描述符
2. **Z/E 分离有效** — MechAware (+5.3% over V2-XGB TSCV) 验证了 Zimmerman-Traxler 机制建模价值
3. **真实天花板 ~0.73** — 非旧报告的 ~0.83（有泄漏）
4. **GNN/Transformer 全部失败** — 1655 样本不足以训练端到端模型
5. **DRFP 看似强大实为泄漏** — 产物立体化学 @/@@ 被直接编码

---

## Historical Results (Legacy, 仅供参考)

> ⚠️ 以下基于旧数据 (1801行) + 旧划分（已确认 group_id 泄漏）。数字虚高。

Top-5 historical (temporal split): V2-XGB 0.783, V2-Stack 0.782, V5-XGB 0.758, DRFP+Cond 0.711, ProtoNet 0.684

详见 `archive/` 中的旧 RESULTS.md。
