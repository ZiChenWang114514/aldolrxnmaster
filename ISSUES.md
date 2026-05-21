# Known Issues & Resolved Items

## ⚠️ 确认的数据泄漏 (2026-05-16)

### LEAK-1: DRFP 标签泄漏 — CONFIRMED

- **机制**: DRFP 编码 `reactants >> product` 的 shingle 差集。产物 SMILES 含 `@`/`@@` 直接编码了立体化学（即标签本身）
- **证据**: 
  - 含立体化学: TSCV = **0.821** (虚高)
  - 去除立体化学: TSCV = **0.577** (真实)
  - 纯反应物: TSCV = **0.250** (随机，证明 DRFP 无法从反应物预测立体)
- **影响**: 旧 DRFP+Cond scaffold=0.826 的"好成绩"完全来自泄漏
- **状态**: 已在 MODEL_REGISTRY 中标为 ⚠️ LEAKAGE，从基准排名中排除

### LEAK-2: 旧 group_id 排序泄漏 — CONFIRMED

- **机制**: `deduplicate.py` 用 `sorted([ketone, aldehyde])` 字母排序，破坏角色区分
- **证据**: scaffold 0.826→0.757 (-7%), grouped 0.807→0.708 (-10%)
- **状态**: V3 已修复 (角色保留 `ketone||aldehyde` key)

---

## 已解决 ✓

| ID | 问题 | 解决方式 | 日期 |
|----|------|---------|------|
| D1 | 35 DRFP 生成失败 | 零向量替代 (影响轻微) | 旧 |
| D3 | Anti-class 不平衡 (C1=10%, C2=7%) | balanced_accuracy + sample_weight | 旧 |
| D4 | 17 行 SA 不一致 | V3 已删除这些行 | 2026-05 |
| M7 | 21 enolate 生成失败 | V3 删除+超时保护 | 2026-05 |
| M8 | Atom count mismatch | Heavy-atom 坐标映射修复 | 2026-05 |
| M11 | 醛基未建模 | aldehyde_steric.py (+11.9%) | 旧 |
| M12 | chirality_valid 用错列 | 改用 Raw_Product_Smiles | 旧 |
| M13 | GNN 全面失败 | 确认为负面结果，归档 | 2026-05 |
| M14 | 单 temporal split 不可靠 | TSCV 4-fold | 2026-05 |
| I2 | 模型 checkpoints 未保存 | 仍未解决，低优先 | — |
| I3 | conda run 缓冲 | --no-capture-output | 2026-05 |

---

## 当前限制 (非 bug，项目固有)

1. **数据量**: 1655 Evans reactions — 不足以训练端到端模型 (GNN/Transformer)
2. **Anti-class 稀少**: C1=10%, C2=7% — 模型在 anti 类上系统性弱
3. **sin_tau1 是近似**: 用基态二面角代理 TS 几何，泛化有上限
4. **构象无 MMFF 优化**: V3 跳过 MMFF 以加速，steric 精度可能略降
5. **真实天花板 ~0.73 TSCV**: 进一步提升需要更多数据或更好的 TS 建模
