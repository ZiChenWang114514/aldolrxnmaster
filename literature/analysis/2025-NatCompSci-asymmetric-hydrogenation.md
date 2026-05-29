文件重命名：s43588-025-00920-8.pdf → 2025-NatCompSci-asymmetric-hydrogenation.pdf

# Chemistry-informed deep learning model for predicting stereoselectivity and absolute configuration in asymmetric hydrogenation

## §0 论文基本信息

| 字段 | 内容 |
|---|---|
| 标题 | Chemistry-informed deep learning model for predicting stereoselectivity and absolute configuration in asymmetric hydrogenation |
| 作者 | Li Cheng, Pan-Lin Shao / Pan-Lin Shao, Guichuan Xing, Bo Zhang |
| 期刊 | Nat. Comput. Sci. 2026, 6, 145-155；online 2025-12-05 |
| DOI | 10.1038/s43588-025-00920-8 |
| 规范化文件名 | 2025-NatCompSci-asymmetric-hydrogenation.pdf |
| 开源代码 / 数据集 | 是；AHO data https://github.com/licheng-xu-echo/AHO.git；code https://github.com/CHENGLi-96/ChemAHNet/releases/tag/ChemAHNet-V1.0 |

## §1 核心贡献

- **问题**：预测 olefin asymmetric hydrogenation 的 stereoselectivity 与 major enantiomer absolute configuration，覆盖 9,478 条反应、1,766 个底物、1,480 个催化剂。
- **方法创新**：ChemAHNet 用 interaction mode 替代直接 R/S 多分类，三模块 MoIM/RCIM/MIM 从 SMILES 学习 moiety、反应组件与 substrate-catalyst interaction。
- **结果**：absolute configuration top-1 accuracy 88.9%，ΔΔG‡ 回归 R2=0.591、RMSE=1.269 kcal/mol；在外部 CPA thiol addition 数据上 R2=0.918。

## §2 建模体系全链路拆解

### 2.1 化学体系定义

- **反应类型**：transition-metal-catalyzed asymmetric olefin hydrogenation。
- **预测目标**：major enantiomer interaction mode/absolute configuration 分类 + ΔΔG‡ 回归。
- **立体化学挑战**：同一 substrate-catalyst interaction mode 可因 CIP priority 变化给出相反 R/S；双 prochiral site 会使 R/S label order 更复杂。
- **与本项目对比**：与 AldolRxnMaster 一样预测绝对构型，且同样面对 CIP priority 与真实反应面选择不一致的问题。

### 2.2 输入表示

| 输入来源 | 表示形式 | 具体参数 | 工具/库 |
|---|---|---|---|
| substrate/ligand/catalyst/solvent | SMILES 字符 token | atom-wise tokenization | PyTorch |
| 机制先验 | interaction mode label | catalyst above substrate = 1, below = 0 | 自定义 |
| 输出标签 | ΔΔG‡ 与 interaction mode | ΔΔG‡ = -RT ln(e.r.)；interaction mode 经机制映射到 absolute configuration | 自定义 |

### 2.3 特征提取管线

```text
Step 1: 文献 AHO 数据 → SMILES standardization 与过滤 → 9,478 unique reactions。
Step 2: SMILES tokens → embedding。
Step 3: MoIM → convolution kernels 3*256, 5*256, 7*256 提取 moiety-level features。
Step 4: RCIM → multihead attention 整合 reaction component information。
Step 5: MIM → max pooling + residual network 建模 substrate-catalyst interaction。
最终: Softmax 输出 interaction mode；linear head 输出 ΔΔG‡。
```

### 2.4 模型选择与架构

- **模型类型**：Chemistry-informed deep learning，含 CNN、attention、residual interaction module。
- **选择理由**：不依赖 DFT/QM 描述符，同时把 AHO 机制显式写进输出空间。
- **关键超参数**：MoIM kernels 3/5/7×256；RCIM multihead attention；MIM max pooling/residual；完整训练细节见 Methods/SI。
- **小数据设计**：对 9,478 条并非小数据；但通过机制标签降低任务复杂度。
- **多输出**：classification + regression。

### 2.5 训练与验证策略

| 要素 | 论文做法 |
|---|---|
| 划分方式 | random 80/10/10 |
| OOD | scaffold/value-range extrapolation；外部 CPA thiol addition 和其他 3 个数据集 |
| 时序泄漏 | 主模型未做时序切分 |
| CV | 主文未作为核心 |
| 类别不平衡 | ΔΔG‡ 分布较均衡 |
| HPO | 论文未在主文详列 |

### 2.6 推理输出与后处理

模型输出 interaction mode 概率与 ΔΔG‡ 连续值；absolute configuration 由 interaction mode 和 CIP priority 规则后处理得到，而不是让模型直接学习 R/S 字符。

### 2.7 关键性能结果

| 指标 | 数值 | 测试集描述 | 原文位置 |
|---|---:|---|---|
| top-1 accuracy | ChemAHNet 88.9% | AHO random 80/10/10 | Fig. 3 |
| baselines | NERF 40.2%, T5Chem 82.4%, Chemformer 86.7%, Molecular Transformer 2.0% | 同上 | Fig. 3 |
| ablation | no MoIM 85.6%, no RCIM 78.8%, no MIM 65.2% | 同上 | Fig. 4 |
| ΔΔG‡ regression | R2=0.591, RMSE=1.269 kcal/mol | AHO | Fig. 4 |
| 外部 CPA | R2=0.918, RMSE=0.209 kcal/mol | Denmark CPA thiol addition 1,075 reactions | Fig. 5 |
| value-range extrapolation | >84% classification accuracy on >97-99% ee tests | extrapolation | Results |

### 2.8 化学价值

ChemAHNet 证明了把“模型预测的对象”从 R/S 绝对标签改成可机制解释的 face/interaction mode，可同时提升立体预测一致性和解释性。

## §3 任务与数据对齐

1. **任务类型**：binary/interaction classification + regression；与 4-class 相似度 5/5。
2. **数据规模与质量**：9,478 条，远大于本项目；文献数据经清洗。
3. **底物空间**：不对称氢化，与醛醇不同；但双前手性中心问题和 Ca/Cb 双中心有概念相似性。
4. **标签定义**：absolute configuration 由机制输出后推，这是本项目最值得借鉴之处。
5. **OOD 评估**：scaffold/value-range/external datasets，明显比当前本项目更完整。

## §4 手性/立体化学处理方式

1. **SMILES 手性编码**：只输入 SMILES，但不把 `@/@@` 识别作为核心；核心是 interaction mode 标签。
2. **CIP vs 相对构型**：明确指出相同机制可能因 CIP priority 得到相反 R/S。
3. **立体中心局部环境**：MoIM 从 functional group/moiety 学局部环境；MIM 专门建 substrate-catalyst interaction。
4. **标签噪声处理**：通过机制中间标签消除 R/S 多分类不一致；没有传统 label smoothing。

## §5 可操作技术建议

#### 建议 A：机制中间标签

| 字段 | 内容 |
|---|---|
| 来自论文 | Fig. 1c,d；interaction mode |
| 论文怎么做 | 模型预测 catalyst-substrate interaction mode，再由 CIP 推 absolute configuration。 |
| 在本项目中怎么用 | 在 `scripts/run_rebuild_v4.py` 增加 `cb_face_label` 或 `aldol_geometry_label`，先预测 enolate/aldehyde face relation，再后处理为 Ca/Cb CIP。 |
| 预期收益 | 避免芳香醛 priority flip 让几何相同样本落入相反 class。 |
| 实现难度 | 高 |
| 优先级 | 高 |

#### 建议 B：交互特征块

| 字段 | 内容 |
|---|---|
| 来自论文 | MIM ablation，移除后 accuracy 降 23.7% |
| 论文怎么做 | 单独模块建模 olefin-catalyst interaction。 |
| 在本项目中怎么用 | 在 `scripts/run_features_v4.py` 新增 aldehyde×auxiliary interaction 列，例如 `ald_pri_priority_proxy * chiral_aux_c4_R`、`ald_Vbur_total_mean * aux_rg_*`、`base_pKa * metal_hardness`。 |
| 预期收益 | 提升 Cb 和 joint 4-class，尤其是条件/底物组合效应。 |
| 实现难度 | 低 |
| 优先级 | 高 |

#### 建议 C：外部任务迁移

| 字段 | 内容 |
|---|---|
| 来自论文 | Fig. 5 CPA 外部数据 R2=0.918 |
| 论文怎么做 | 同一架构迁移到机制不同的 asymmetric catalysis。 |
| 在本项目中怎么用 | 在 `scripts/run_protonet.py` 或新增 meta learner 中，将 Evans 作为 source task，Crimmins/Oppolzer/Myers 作为 target tasks。 |
| 预期收益 | 改善非 Evans 小子集泛化。 |
| 实现难度 | 中 |
| 优先级 | 中 |

| 建议 | 预期收益 | 实现难度 | 优先级 |
|---|---|---|---|
| A | 根治 CIP/机制标签错位 | 高 | 高 |
| B | 捕获组合效应 | 低 | 高 |
| C | 非 Evans 泛化 | 中 | 中 |

## §6 写作借鉴

- 可作为本项目“absolute configuration 不能直接作为 naive string/class label”的高影响力引用。
- 其 ablation 叙事很强：每个化学先验模块都要有 drop 数字。本项目应对 AldPriority、chiralenv、condition、steric 做同样 ablation。
- 需要谨慎：其 random split 可能比本项目 TSCV 宽松，因此引用性能时要强调任务与划分不同。

## §7 综合评分

| 维度 | 评分 | 理由 |
|---|---:|---|
| 建模体系相关性 | 5 | 同为 absolute configuration |
| 特征工程启示 | 4 | interaction module 思路可迁移 |
| 手性处理启示 | 5 | 明确解决 CIP/机制错位 |
| 小数据策略 | 3 | 数据量较大 |
| 写作/叙事借鉴 | 5 | Nature 子刊结构非常适合对标 |
| 综合优先级 | 5 | 阶段1核心引用 |

**一句话总结**：ChemAHNet 给本项目最直接的启示是：不要让模型直接背 CIP 标签，应先预测反应机制/几何面选择，再映射到 R/S。
