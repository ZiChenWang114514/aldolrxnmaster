文件重命名：s41467-024-45102-8.pdf → 2024-NatCommun-chirality-transformers.pdf

# Difficulty in chirality recognition for Transformer architectures learning chemical structures from string representations

## §0 论文基本信息

| 字段 | 内容 |
|---|---|
| 标题 | Difficulty in chirality recognition for Transformer architectures learning chemical structures from string representations |
| 作者 | Yasuhiro Yoshikai / Hiroyuki Kusuhara |
| 期刊 | Nat. Commun. 2024, 15, 1243 |
| DOI | 10.1038/s41467-024-45102-8 |
| 规范化文件名 | 2024-NatCommun-chirality-transformers.pdf |
| 开源代码 / 数据集 | 是；https://github.com/mizuno-group/ChiralityMisunderstanding |

## §1 核心贡献

- **问题**：研究 Transformer 在 randomized SMILES → canonical SMILES 学习中到底何时学会分子结构，尤其是 chirality token。
- **方法创新**：按训练步数分析 perfect/partial accuracy、fingerprint similarity、MoleculeNet 下游性能和 token-wise masked accuracy。
- **结果**：Transformer 很快学会局部结构，但对 `@/@@` chirality token 学习很慢且会长期停滞；pre-LN 可缓解停滞。

## §2 建模体系全链路拆解

### 2.1 化学体系定义

- **任务类型**：SMILES-to-SMILES canonicalization / structure recognition，不是化学反应。
- **预测目标**：给定 randomized SMILES，输出 canonical SMILES；重点观察完整 accuracy 与 chiral token accuracy。
- **立体化学挑战**：`@/@@` 在字符串中是局部符号，但其含义依赖邻接顺序；Transformer 可记住连接关系却不稳定学习手性。
- **与本项目对比**：AldolRxnMaster 若使用 Chemformer/T5Chem 类字符串模型，普通 accuracy 可能掩盖 Ca/Cb stereochemistry failure。

### 2.2 输入表示

| 输入来源 | 表示形式 | 具体参数 | 工具/库 |
|---|---|---|---|
| molecules | randomized SMILES | ZINC-15；heavy atoms 3-50；元素 H/B/C/N/O/F/P/S/Cl/Br/I | RDKit/ZINC |
| target | canonical SMILES | 保留 chirality tokens | RDKit |
| model input | token sequence | Transformer encoder-decoder；embedding 512，FFN 2048 | Transformer |
| 手性信息 | `@` / `@@` tokens | 不额外注入 CIP/3D，只靠字符串 | SMILES |

### 2.3 特征提取管线

```text
Step 1: ZINC-15 molecules → 过滤元素与 heavy atom count。
Step 2: 生成 randomized SMILES 输入与 canonical SMILES 目标。
Step 3: tokenization → Transformer training，25,000 tokens/step。
Step 4: 输出 SMILES → exact match / partial structure / chiral token accuracy。
最终: 分析一般结构 token 与 chirality token 的学习速度差异。
```

### 2.4 模型选择与架构

- **模型类型**：Transformer；比较 Pre-LN 等架构变体。
- **选择理由**：Transformer 是 SMILES 反应预测/性质预测常用骨架。
- **关键超参数**：embedding 512、feed-forward 2048、最多 80,000 steps；测试不同初始化、optimizer 与 chirality frequency。
- **小数据设计**：不是小数据任务；重点在学习困难机制。
- **多任务/多输出**：序列生成，额外按 token 类型评估。

### 2.5 训练与验证策略

| 要素 | 论文做法 |
|---|---|
| 划分方式 | ZINC-15 训练/测试，约 9,057 test molecules |
| 底物外推 | 非反应外推 |
| 时序泄漏 | 不相关 |
| CV | 多初始化/训练步数比较 |
| 类别不平衡 | 尝试提高 chirality token frequency |
| HPO | 架构与优化器变体比较 |

### 2.6 推理输出与后处理

输出 canonical SMILES；评估不仅看 exact match，也看 partial structure、MACCS/ECFP Tanimoto 与 chiral token accuracy。

### 2.7 关键性能结果

| 指标 | 数值 | 测试集描述 | 原文位置 |
|---|---:|---|---|
| test size | ~9,057 molecules | ZINC-15 subset | Methods |
| 训练步数 | up to 80,000 | Transformer canonicalization | Results |
| failure mode | perfect accuracy 可在 10k-70k steps 停滞 | 多初始化 | Results |
| chiral token | `@/@@` accuracy 明显低于一般 token | token-level analysis | Results |

### 2.8 化学价值

该文直接否定“把手性 SMILES 喂给 Transformer 就等于模型理解手性”的朴素假设。对本项目，任何字符串深度模型都必须配套显式立体标签测试。

## §3 任务与数据对齐

- **任务类型**：表示学习诊断，不是反应预测；与本项目相似度 3/5。
- **数据**：ZINC-15，训练 randomized/canonical SMILES；约 9,057 molecules test；最多训练 80,000 steps。
- **输入/模型**：Transformer，embedding 512，feed-forward 2048，encoder/decoder layers 按原始 Transformer；25,000 tokens/step。
- **指标**：perfect accuracy、partial accuracy、MACCS/ECFP Tanimoto、MoleculeNet RMSE/AUC、masked token accuracy。
- **关键发现**：partial accuracy 很快到 1.0；perfect accuracy 可在 10,000-70,000 steps 维持约 0.6 后才跃升；`@/@@` 混淆是主要原因。

## §4 手性/立体化学处理方式

- **SMILES 手性编码**：核心结论是标准 Transformer 对 `@/@@` 学习困难，甚至在 overall accuracy 很高时 chiral token accuracy 仍低。
- **CIP vs 相对构型**：不讨论反应 CIP 标签，但证明字符串模型不能可靠捕获 chirality。
- **局部环境**：Transformer 先学局部 fingerprints，后学整体连接与 chirality。
- **标签噪声**：不是标签噪声论文；关注模型认知失败。

## §5 可操作技术建议

#### 建议 A：禁用纯SMILES基线

| 字段 | 内容 |
|---|---|
| 来自论文 | Fig. 4 chirality token difficulty |
| 论文怎么做 | 证明 `@/@@` token 是 Transformer 长期停滞的主要来源。 |
| 在本项目中怎么用 | 在论文和 `scripts/run_chemprop.py`/Transformer baseline 报告中明确：纯 SMILES 序列模型只能作为负面对照，不能作为主模型。 |
| 预期收益 | 为 Morgan + 显式 chirality/AldPriority 特征提供文献依据。 |
| 实现难度 | 低 |
| 优先级 | 高 |

#### 建议 B：手性token审计

| 字段 | 内容 |
|---|---|
| 来自论文 | token-wise masked accuracy |
| 论文怎么做 | 单独统计 `@/@@` 预测错误。 |
| 在本项目中怎么用 | 在 `scripts/run_data_audit.py` 增加 reactant/product SMILES 中 `@/@@`、RDKit chiral centers、CIP labels 的一致性审计。 |
| 预期收益 | 提前发现手性 SMILES 缺失或反转样本。 |
| 实现难度 | 低 |
| 优先级 | 高 |

#### 建议 C：显式优先级特征

| 字段 | 内容 |
|---|---|
| 来自论文 | 结论 3：chirality 需额外结构或任务辅助 |
| 论文怎么做 | 建议加入教模型 chirality 的结构/辅助任务。 |
| 在本项目中怎么用 | 继续扩展 `data/features_v4/v4_features.csv` 的 `ald_pri_*`：加入 `ald_pri_second_atomic_num`、`ald_pri_aromatic_ipso_atomic_num`、`ald_pri_cip_flip_risk`。 |
| 预期收益 | Cb 预测和 class 0/3 改善。 |
| 实现难度 | 中 |
| 优先级 | 高 |

| 建议 | 预期收益 | 实现难度 | 优先级 |
|---|---|---|---|
| A | 方法选择证据 | 低 | 高 |
| B | 数据质量 | 低 | 高 |
| C | Cb 特征增强 | 中 | 高 |

## §6 写作借鉴

1. **故事结构**：用一个简单自监督任务揭示广泛模型假设的漏洞；本项目可类似展示 CIP 标签与几何标签的反例。
2. **方法论证**：不仅报告 overall accuracy，而要拆 token/type/subset；AldolRxnMaster 应拆 Ca、Cb、class 0/3、芳香醛子集。
3. **Benchmark 严格性**：该文说明常规下游 benchmark 不足以验证手性；本项目需要专门手性 stress test。
4. **新文献线索**：SMILES Transformer、Chemformer、T5Chem 以及 chirality-aware molecular representation 文献都可纳入“通用模型不足”段落。

## §7 综合评分

| 维度 | 评分 | 理由 |
|---|---:|---|
| 建模体系相关性 | 3 | 非反应预测 |
| 特征工程启示 | 4 | 支持显式手性特征 |
| 手性处理启示 | 5 | 直接证明字符串手性弱点 |
| 小数据策略 | 2 | 与小数据无关 |
| 写作/叙事借鉴 | 4 | 可作为负面证据 |
| 综合优先级 | 4 | 必引，少量代码审计可实现 |

**一句话总结**：这篇论文是本项目拒绝依赖纯 SMILES Transformer、坚持显式 CIP/priority/chirality 特征的最强证据。
