# Lessons Learned — AldolRxnMaster

## L10: CIP R/S ≠ syn/anti — 绝对构型无法推断相对构型 (2026-05-27)

**问题**: `label_SA = int(Ca==Cb)` 作为 syn/anti 启发式，准确率仅 ~52%（等同掷硬币）。基于 Ca CIP + 辅基构型的 `label_syn_anti` 同样不可靠（~52%）。
**根因**: CIP R/S 是**绝对构型**（依赖取代基优先级规则），syn/anti 是**相对构型**（两个立体中心的空间关系）。同一 syn 产物在不同底物上可以是 (R,S) 或 (S,S)，因为：
- **Cb CIP 翻转**: 芳香醛（如 PhCHO）使 Cb 的 CIP 优先级翻转（芳环 > 烷基链）
- **Ca CIP 翻转**: 酮 α-取代基含杂原子（如 OBn, OTBS）时 Ca 的 CIP 优先级翻转
- 两个中心的 CIP 码都不稳定，任何基于 CIP 推断 syn/anti 的组合都不可靠

**实证数据** (2334 行 substrate-controlled aldol):
- `label_SA` vs `label_syn_anti_3d` 一致率: **45.6%**（比随机还差，因为 CIP 翻转与 syn/anti 无关）
- 二面角分布: 清晰双峰（~±60° gauche=syn, ~±180° anti-periplanar=anti）
- 3D syn 比例: 62.6% syn / 36.1% anti / 1.3% 失败
- Evans 子集: syn 63.4%（符合 Evans Z(O)-enolate Zimmerman-Traxler TS 预期）
- 辅基分布: 所有 6 种辅基均有 syn/anti 清晰分离

**迭代过程**:
1. 第 1 轮: 尝试 `label_SA = int(Ca==Cb)` → 52% 准确率
2. 第 2 轮: 尝试 Ca CIP + 辅基 4R/4S 构型(SMARTS `useChirality=True`) → 仍 52%
3. 第 3 轮: 分析发现 Ca CIP 也会因取代基杂原子翻转 → 确认 CIP 路线不可行
4. 第 4 轮(最终): 3D 二面角法 — ETKDGv3+MMFF 生成构象 → OH-Cb-Ca-C(=O) 二面角 → |θ|<90°=syn → **98.7% 成功率**

**修复**: 新增 `step08b_3d_synanti.py`，删除 step08 中所有 `aux_config`/`label_syn_anti` 相关代码。输出 4 列: `label_syn_anti_3d`, `dihedral_oh_cb_ca_co`, `conformer_energy`, `synanti_confidence`。

**教训推广**: 任何基于 CIP R/S 推断相对构型（syn/anti, endo/exo, cis/trans, E/Z）的方法都不可靠。CIP 优先级规则是人为约定（基于原子序数/质量），不反映空间关系。3D 坐标（二面角、距离）是确定相对构型的唯一正确途径。

---

## L9: DRFP 标签泄漏 — 反应指纹对立体化学预测的陷阱 (2026-05-16)

**问题**: DRFP TSCV=0.87 远超所有其他模型 (次好 0.73)。
**根因**: DRFP 编码 `reactants >> product` 的 shingle 差集。产物 SMILES 含 @/@@ 直接编码了立体化学——即我们预测的标签本身。模型不学化学，直接"读答案"。
**证据**: 
- 含 @/@@: 0.821
- 去除 @/@@: 0.577 (真实水平，远低于 MechAware 0.733)
- 纯反应物: 0.250 (随机)
**教训**: 任何使用产物信息的反应表征（DRFP、RXNFP、reaction SMILES 模型）在立体化学预测任务中都可能有泄漏。必须验证：如果模型性能显著超过基于反应物+条件的模型，首先怀疑泄漏。
**推广**: 对于任何"预测产物性质"的任务，如果输入包含产物本身的信息，都是泄漏。

---

## V3 Rebuild Lessons



## L1: RDKit MMFF 线程死锁 (2026-05-15)

**问题**: Step 10 构象生成跑了 4+ 小时卡死在第一个分子。
**根因**: `AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=1)` 在某些分子上 RDKit 内部仍会创建 64 个线程，导致 MMFF 优化死锁。内存仅 454MB 证实几乎未处理任何分子。
**修复**: 
- 设置 `params.numThreads = 0`（单线程嵌入）
- 添加 `signal.alarm(60)` 超时保护，每分子最多 60 秒
- 添加 `maxIters=200` 限制 MMFF 优化步数
**教训**: 对长时间运行的计算任务，必须有超时+进度日志+内存监控。

## L2: 构象生成未去重 (2026-05-15)

**问题**: 4263 行数据中 unique 分子仅 4378 个，但代码对每行重新计算 → 8526 次嵌入。
**根因**: `step10_conformers.py` 的 for 循环逐行处理，未先提取 unique SMILES。
**修复**: 先 `unique()` 去重，算完后用 dict 映射回行。速度提升约 2 倍。
**教训**: 构象生成是最贵的操作，必须去重后再计算。

## L3: `ketone_to_enolate()` 参数错误 (2026-05-15)

**问题**: 调用 `ketone_to_enolate(smi, base=base)` 但该函数只接受 1 个参数。
**根因**: 写代码时假设函数有 `base` 参数，未查接口。TypeError 被 except 静默吞掉。
**修复**: 删除 `base=base` 参数。
**教训**: 调用现有函数前必须确认签名。静默 except 可以掩盖 bug。

## L4: CIP R/S 不能全局映射到 Ca/Cb 标签 (2026-05-15)

**问题**: Step 4 CIP 校验用全局多数投票确定 Ca=0 → R/S 映射，结果只有 52% vs 48%，删掉了 74% 的数据 (3523/4750)。
**根因**: syn/anti 是相对立体化学（zig-zag 投影定义），不等同于 "Ca/Cb same R/S"。CIP R/S 依赖底物取代基优先级，同一 syn 产物在不同底物上可以是 (R,S) 或 (S,R)。
**修复**: 改为 SA 一致性校验（label_SA vs Ca==Cb），只删不一致行（~215 行，非 3523 行）。
**教训**: 有机化学中 R/S 是绝对构型，syn/anti 是相对构型，两者不能直接映射。

## L5: RDKit SetBondStereo + MolToSmiles 丢失立体信息 (2026-05-15)

**问题**: 用 `bond.SetStereo(STEREOZ)` + `bond.SetStereoAtoms()` 后，`MolToSmiles` 输出不含 `/\` 标记，Z 和 E SMILES 完全相同。
**根因**: RDKit 的 MolToSmiles 在规范化过程中可能丢弃它认为"不可靠"的双键立体信息，特别是对于通过 API 手动设置（而非从 SMILES 解析或 3D 推导）的立体化学。
**修复**: 改用 3D 方法——生成不指定 Z/E 的烯醇盐构象，然后用二面角测量 (|θ| < 90° = Z, ≥ 90° = E) 将构象分类到 Z 和 E 子集。
**教训**: RDKit 的 2D 立体化学操作不如 3D 方法可靠。对于需要精确控制的立体化学，优先使用 3D 坐标。

## L7: step11 steric 坐标维度不匹配 (2026-05-15)

**问题**: conformer ensemble 存的是 AddHs 后的 mol (35 atoms) + 全 coords (35,3)，但 steric_descriptors 的 `compute_single_conformer_descriptors` 期望 no-H mol (19 atoms) + heavy-atom coords (19,3)。
**根因**: step10 用 `Chem.AddHs(mol)` 后嵌入 3D，coords 包含 H 原子。step11 直接传给 steric 函数导致 atom count mismatch → 返回 None → 所有行 steric 失败。
**修复**: 在 step11 中添加 `_extract_heavy_coords()` 函数，嵌入前提取 heavy atom indices，传给 steric 函数时只用 heavy atom 坐标。
**教训**: 跨模块传递分子数据时，必须确认 H 原子处理一致（AddHs/RemoveHs）。

## L8: 代码审计发现的系统性问题 (2026-05-15)

**批次修复**: 34 个潜在 bug 通过深度审计发现，已修复 7 个关键问题:
1. Z/E 二面角可能测量到 H 原子 → 强制只用 heavy atom neighbor
2. MMFF fallback 不清理旧构象 → 添加 `RemoveAllConformers()`
3. NaN hash 碰撞 → 用 `<NA>` 替代 `str(NaN)`
4. 未知金属全零特征 → 映射到 "unknown"
5. V3 indices 重映射硬编码 → 用 interim CSV 重建行序
6. 死代码参数 `use_enolate_smarts` → 删除
7. step11 heavy-atom 坐标映射 → 添加 `_extract_heavy_coords()`

**教训**: 写完代码后必须做交叉模块审计，特别是数据在模块间传递时的格式假设 (H atom/no-H, index alignment, NaN handling)。

## L6: `conda run` 缓冲 stdout (已知)

**问题**: `conda run` 缓冲 stdout，导致 `.output` 文件或 tee 管道为空。
**修复**: 用 `--no-capture-output` 标志，或直接写入 log 文件（logging 模块写 FileHandler）。
**教训**: 后台任务必须用独立 log 文件，不要依赖 stdout 重定向。
