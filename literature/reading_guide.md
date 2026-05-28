# AldolRxnMaster 文献阅读流程指导

> **目标**：系统消化 `sota_stereo_prediction_2022_2026.md` 中的 43 篇文献，提取对项目的具体技术启示。
>
> **配套工具**：`prompt.md`（逐篇分析 agent prompt）
>
> **输出目录**：`literature/analysis/[编号]_[短标题].md`（每篇一个分析文件）

---

## 阅读策略总览

| 阶段 | 篇数 | 方式 | 产出 |
|------|------|------|------|
| **阶段 1：精读核心** | 5 篇（4★） | PDF 全文 + 完整 prompt | 9节完整分析，≥3条建议 |
| **阶段 2：方法提取** | 8 篇（3★精选） | PDF 全文 + prompt §3/§5/§7重点 | 特征/模型/手性处理建议 |
| **阶段 3：摘要快扫** | 30 篇（1-2★） | 摘要+引言 + 只回答 §7/§9 | 快速判断是否升级处理 |

---

## 阶段 1：精读核心（5 篇，4★）

> 这 5 篇与本项目任务结构最接近，或是写作时必须引用的高影响力文献。**每篇独立开一个 agent 会话**，将 `prompt.md` 全文 + 论文全文一起送入。

---

### P1-1｜Predicting glycosylation stereoselectivity using machine learning

- **作者**：Sooyeon Moon, Peter H. Seeberger, Kerry Gilmore 等 (Max-Planck Institute)
- **期刊**：*Chemical Science* 2021
- **链接**：[https://doi.org/10.1039/d0sc06222g](https://doi.org/10.1039/d0sc06222g)
- **为何必读**：最接近本项目的先驱工作——同为底物控制立体选择性预测，268条小数据，随机森林+QM描述符，"相同底物骨架+不同条件"场景与本项目完全平行
- **重点关注**：§3特征工程（QM描述符10维），§4如何处理小数据，§7能否迁移其描述符设计

---

### P1-2｜Anomeric Selectivity of Glycosylations through a Machine Learning Lens

- **作者**：Sooyeon Moon, Sourav Chatterjee, Peter H. Seeberger, Kerry Gilmore (Max-Planck Institute)
- **期刊**：*Journal of the American Chemical Society* 2025, 147(40)
- **链接**：[https://doi.org/10.1021/jacs.5c07561](https://doi.org/10.1021/jacs.5c07561)
- **为何必读**：P1-1 的升级版，数据量扩至 >10,000 条，引入多任务ML（同时预测主异头体/次异头体/比例）——本项目未来扩展路径的直接参考
- **重点关注**：§4多任务学习架构，§2数据集构建策略，§8论文故事结构

---

### P1-3｜Chemistry-informed deep learning model for predicting stereoselectivity in asymmetric hydrogenation

- **作者**：Qi Yang 等
- **期刊**：*Nature Computational Science* 2025
- **链接**：[https://doi.org/10.1038/s43588-025-00920-8](https://doi.org/10.1038/s43588-025-00920-8)
- **为何必读**：Nature 子刊，化学先验知识嵌入深度学习，处理绝对构型（R/S）预测——与本项目标签类型完全相同
- **重点关注**：§5手性处理方式（如何嵌入化学先验），§4模型架构（DL vs GBM 在小数据下的取舍），§8写作框架

---

### P1-4｜Data-Driven Prediction of Enantioselectivity for the Sharpless Asymmetric Dihydroxylation

- **作者**：相关组
- **期刊**：*ACS Central Science* 2025
- **链接**：[https://doi.org/10.1021/acscentsci.5c00900](https://doi.org/10.1021/acscentsci.5c00900)
- **为何必读**：ACS Cent Sci，数据驱动+实验验证闭环，Sharpless AD 是经典手性辅基/催化剂控制反应，故事结构（预测→合成验证）是本项目论文投稿时最可能需要效仿的范式
- **重点关注**：§6性能对标，§8实验验证部分的论文写作方式，§7数据集外推测试设计

---

### P1-5｜Connecting the complexity of stereoselective synthesis to the evolution of predictive tools（综述）

- **作者**：相关组
- **期刊**：*Chemical Science* 2025
- **链接**：[https://doi.org/10.1039/d4sc07461k](https://doi.org/10.1039/d4sc07461k)
- **为何必读**：高引综述，系统梳理立体选择性合成预测工具的历史演化——本项目 Introduction 段落写作的框架来源，必须引用
- **重点关注**：§1综述如何组织叙事，§8写作借鉴（Introduction引言段落结构），§7综述提到的未解决问题是否与本项目填补的空白一致

---

## 阶段 2：方法提取（8 篇，3★ 精选）

> 重点提取 **特征工程**、**小数据策略**、**手性处理** 三个方面。送入 agent 时可只要求回答 `prompt.md` 的 §3、§5、§7 三节，其余简答即可。

---

### P2-1｜Leveraging Limited Experimental Data with Machine Learning: CBS Reduction

- **期刊**：*Journal of the American Chemical Society* 2024
- **链接**：[https://doi.org/10.1021/jacs.4c01286](https://doi.org/10.1021/jacs.4c01286)
- **关注点**：专门针对 **小数据 ML**（CBS 不对称还原，数据量与本项目同级），Bayesian/GP方法在<500条数据下的表现 vs GBM

---

### P2-2｜Bayesian Meta-Learning for Few-Shot Reaction Outcome Prediction of Asymmetric Hydrogenation

- **期刊**：*Angewandte Chemie International Edition* 2025
- **链接**：[https://doi.org/10.1002/anie.202503821](https://doi.org/10.1002/anie.202503821)
- **关注点**：**few-shot 元学习**在不对称氢化上的应用，当前 Evans-only 训练集 1654 条，Crimmins/Oppolzer 各只有数百条——元学习能否跨辅基泛化？

---

### P2-3｜A meta-learning approach for selectivity prediction in asymmetric catalysis

- **期刊**：*Nature Communications* 2025
- **链接**：[https://doi.org/10.1038/s41467-025-58854-8](https://doi.org/10.1038/s41467-025-58854-8)
- **关注点**：Nat Commun，元学习框架+跨反应类型迁移，直接参考如何把 Evans 模型迁移到 Crimmins/Oppolzer

---

### P2-4｜Difficulty in Chirality Recognition for Transformer Architectures Learning Chemical Structures from String Representations

- **期刊**：*Nature Communications* 2024
- **链接**：[https://www.nature.com/articles/s41467-024-45102-8](https://www.nature.com/articles/s41467-024-45102-8)
- **关注点**：**Transformer 对 SMILES 手性编码的识别失效**——直接支撑本项目放弃纯 SMILES 序列模型、保留 Morgan指纹+手性特征的设计决策；写作时作为负面证据引用

---

### P2-5｜Stereoisomers Are Not Machine Learning's Best Friends

- **期刊**：*Journal of Chemical Information and Modeling* 2024
- **链接**：[https://doi.org/10.1021/acs.jcim.4c00318](https://doi.org/10.1021/acs.jcim.4c00318)
- **关注点**：系统证明标准 ML 管线对立体异构体的盲点，可作为本项目 Introduction 的直接动机引用；关注其提出的改进方向

---

### P2-6｜Predicting the stereoselectivity of chemical reactions by composite machine learning method

- **期刊**：*Scientific Reports* 2024
- **链接**：[https://doi.org/10.1038/s41598-024-62158-0](https://doi.org/10.1038/s41598-024-62158-0)
- **关注点**：**复合ML方法**（ensemble+stacking），多类型特征融合策略，类别不平衡处理

---

### P2-7｜Evaluating Predictive Accuracy in Asymmetric Catalysis: A Machine Learning Perspective on Local Reaction Space

- **期刊**：*ACS Catalysis* 2025
- **链接**：[https://doi.org/10.1021/acscatal.5c01051](https://doi.org/10.1021/acscatal.5c01051)
- **关注点**：**局部反应空间** vs 全局泛化，本项目 Evans-only TSCV=0.795 vs 全数据集 0.624 的差距正好是这个问题的体现；提供理论框架

---

### P2-8｜Data-Efficient, Chemistry-Aware Machine Learning Predictions of Diels–Alder Reaction Outcomes

- **期刊**：*Journal of the American Chemical Society* 2024
- **链接**：[https://doi.org/10.1021/jacs.4c03131](https://doi.org/10.1021/jacs.4c03131)
- **关注点**：**数据高效**的化学感知ML，Diels-Alder 也是经典有机立体化学反应；关注如何用化学约束减少所需数据量

---

## 阶段 3：摘要快扫（30 篇，1-2★）

> 只送入摘要+引言（通常200-400词），让 agent 只回答 `prompt.md` §7（建议）和 §9（总评）。快速判断是否需要升级到阶段2。

---

### 3★ 其余待快扫

| 编号 | 标题 | 链接 |
|------|------|------|
| [4] | Artificial Intelligence for Predicting Stereoselectivity in Glycosylation | [链接](https://doi.org/10.31635/ccschem.025.202506735) |
| [3] | ScopeMap: AI-Assisted Workflow for Mapping Reaction Scope | [链接](https://doi.org/10.1002/anie.2455429) |
| [5] | Enantioselectivity prediction: pallada-electrocatalysed C–H activation | [链接](https://doi.org/10.1038/s44160-022-00233-y) |
| [8] | Feed-Forward NN for Asymmetric Negishi Reaction | [链接](https://doi.org/10.1021/acscentsci.3c00512) |
| [15] | ML for asymmetric hydrogenation catalyst discovery | [链接](https://doi.org/10.1039/d4sc03647f) |
| [26] | GNN for Data Checking of Asymmetric Catalysis Literature | [链接](https://doi.org/10.3390/molecules30020355) |
| [39] | %VBur steric maps: from predictive catalysis to ML | [链接](https://doi.org/10.1039/D3CS00725A) |
| [40] | ML-guided strategies for reaction conditions design | [链接](https://doi.org/10.3762/bjoc.20.212) |

---

### 2★ 待快扫

| 编号 | 标题 | 链接 |
|------|------|------|
| [6] | Data Science for Chiral Phosphoric Acid Catalysts | [链接](https://doi.org/10.1016/j.chempr.2023.02.020) |
| [7] | Data-Driven Workflow for Generality in Asymmetric Catalysis | [链接](https://doi.org/10.1021/jacs.3c03989) |
| [9] | Predicting Enantioselective Catalysts via Tunable Fragment Descriptors | [链接](https://doi.org/10.1002/anie.202218659) |
| [12] | Data Science for Stereoconvergent Nickel-Catalyzed Reduction | [链接](https://doi.org/10.1021/acscatal.4c00650) |
| [20] | Generality-Driven Optimization of Enantio- and Regioselective Reduction | [链接](https://doi.org/10.1002/anie.202519425) |
| [23] | DL for asymmetric β-C–H bond activation | [链接](https://doi.org/10.1039/D2DD00084A) |
| [24] | ML for Enantioselective C–H Activation: Generative AI to Experiment | [链接](https://doi.org/10.1039/d5sc01098e) |
| [27] | Homogeneous Catalyst GNN for Ligand Optimization | [链接](https://doi.org/10.1016/j.isci.2025.111881) |
| [28] | ChIRo: 3D Chirality Representations Invariant to Bond Rotations | [链接](https://openreview.net/forum?id=hm2tNDdgaFK) |
| [29] | ChiENN: Molecular Chirality with Graph Neural Networks | [链接](https://link.springer.com/chapter/10.1007/978-3-031-43418-1_3) |
| [30] | 3DReact: Geometric Deep Learning for Chemical Reactions | [链接](https://doi.org/10.1021/acs.jcim.4c00104) |
| [34] | Stereoelectronics-Infused Molecular Graphs (SIMG) | [链接](https://www.nature.com/articles/s42256-025-01031-9) |
| [35] | Chiral Determinant Kernels (ChiDeK) | [链接](https://arxiv.org/abs/2602.07415) |
| [37] | Chirality Descriptors from SMILES Heteroencoders | [链接](https://link.springer.com/article/10.1186/s13321-025-01080-7) |
| [41] | ML Strategies for Reaction Development: Low-Data Limit | [链接](https://doi.org/10.1021/acs.jcim.3c00577) |
| [42] | Rethinking chemical research in the age of LLMs | [链接](https://doi.org/10.1038/s43588-025-00811-y) |
| [43] | Foundation Models in Chemistry (Perspective) | [链接](https://doi.org/10.1021/jacsau.4c01160) |

---

### 1★ 可选（仅在写作查引用时翻阅）

| 编号 | 标题 | 链接 |
|------|------|------|
| [13] | ML for Magnesium-Catalyzed Asymmetric Reactions | [链接](https://doi.org/10.1002/anie.202318487) |
| [22] | ML for amidase enantioselectivity and variant design | [链接](https://doi.org/10.1038/s41467-024-53048-0) |
| [32] | Root-Aligned SMILES for Chemical Reaction Prediction | [链接](https://doi.org/10.1039/d2sc02763a) |
| [33] | fragSMILES: Advanced Chirality & Fragment Representation | [链接](https://pubs.rsc.org/en/content/articlelanding/2025/cc/d5cc02641e) |
| [38] | Chirality-Focused GNN for Liver Microsome Clearance | [链接](https://doi.org/10.1021/acs.jcim.4c00243) |

---

## 操作步骤

### 下载论文

```bash
# 学校 VPN 已开时，直接点上方链接下载 PDF
# 无法访问时，在浏览器地址栏输入：
# sci-hub.se/[DOI]
# 例：sci-hub.se/10.1039/d0sc06222g

# arXiv 论文直接下载：
wget https://arxiv.org/pdf/2602.07415 -O literature/pdfs/35_ChiDeK.pdf
```

### 运行 agent 分析

```
1. 打开新的 Claude Code 对话
2. 粘贴 literature/prompt.md 全文
3. 粘贴论文全文（或上传 PDF）
4. 指令："请按 prompt 的9节结构分析这篇论文"
5. 将输出保存到 literature/analysis/[编号]_[短标题].md
```

### 提交分析结果

```bash
git add literature/analysis/
git commit -m "lit: analysis [编号] [短标题]"
git push
```

---

## 优先级决策树

```
这篇论文是否处理 R/S 绝对构型预测？
├── 是 → 进入阶段1精读
└── 否 → 数据量 < 2000 条？
         ├── 是 → 进入阶段2方法提取
         └── 否 → 直接阶段3摘要快扫
```

---

## 预期产出

完成3个阶段后，`literature/analysis/` 应包含：
- **5 个完整分析**（阶段1，9节全填）
- **8 个方法摘要**（阶段2，§3/§5/§7）
- **30 个快扫记录**（阶段3，§7/§9）

最终输出：**一份针对 AldolRxnMaster 的技术建议汇总**，直接指导下一轮建模迭代。

---

*指导文档版本：2026-05-28 | 配套文件：sota_stereo_prediction_2022_2026.md, prompt.md*
