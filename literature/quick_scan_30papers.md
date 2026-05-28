# 30篇文献快扫摘要（阶段3）

> 已精读13篇（阶段1+2：[1],[2],[10],[11],[14],[16],[17],[18],[19],[21],[25],[31],[36]）。
> 本文档覆盖剩余30篇的摘要级快扫，按原编号排列。

---

## [3] ScopeMap: AI-Assisted Workflow for Mapping Reaction Scope
- **体系**: 仿生醛醇反应 + 钴催化偶联
- **任务**: 分类（反应性/非反应性边界映射）
- **数据量**: <4% 底物空间用于训练
- **输入表示**: 分子描述符 + 修正CVT算法几何表示
- **模型**: CVT + 动态几何排斥势（非传统ML）
- **关键性能**: F1 > 90%
- **启示**: CVT/U-Score 可量化我们1654条数据在底物空间的覆盖度偏差；HITL主动学习可指导未来实验扩展

## [4] AI for Predicting Stereoselectivity in Glycosylation（综述）
- **体系**: 糖苷化反应 alpha/beta 异头选择性
- **任务**: 回归（alpha/beta比值）+ 产率优化
- **数据量**: ~100条（Milo组）；综述汇总多种策略
- **输入表示**: DFT描述符 / SMILES / 反应条件编码
- **模型**: RF / 决策树 / 迁移学习 / 贝叶斯优化
- **关键性能**: RF RMSE=6.8
- **启示**: RF+DFT描述符在~100条小数据上有效，支持ExtraTrees路线；迁移学习可从Evans→非Evans辅基

## [5] Enantioselectivity of pallada-electrocatalysed C–H activation via TS knowledge
- **体系**: Pd电催化不对称C-H活化
- **任务**: 回归（ee%）
- **数据量**: 实验组合 + 虚拟评估空间846,720种
- **输入表示**: **DFT过渡态描述符**（TS几何+电子性质）
- **模型**: RF / XGBoost
- **关键性能**: TS描述符外推能力远优于通用描述符
- **启示**: **核心验证**——TS描述符>>通用描述符，B1-xTB路线正确；XGB应与ET并行benchmark

## [6] Data Science for Chiral Phosphoric Acid Catalysts
- **体系**: 手性磷酸催化转移氢化/环脱水/去对称化
- **任务**: 分类 + 回归（ee）
- **数据量**: 虚拟库>300K → 训练集仅**20个催化剂**
- **输入表示**: **91个DFT描述符**（%VBur/NBO/ChelpG/Hirshfeld）
- **模型**: 单变量分类树 + 多元线性回归（2-3参数）
- **关键性能**: R²=0.98, MAE=0.08 kcal/mol
- **启示**: **极小数据+高质量DFT描述符=超强外推**；单变量分类树可解释性策略值得借鉴

## [7] Data-Driven Workflow for Generality in Asymmetric Catalysis
- **体系**: 有机催化Mannich反应 + CPA催化亚胺加成
- **任务**: 无监督聚类 + 回归（催化剂通用性量化）
- **数据量**: 文献策展数据集
- **输入表示**: 反应描述符（催化剂结构+底物特征）
- **模型**: PCA+k-means + 回归 + "designer equations"
- **关键性能**: 重点在通用性量化方法而非传统精度
- **启示**: "通用性"框架可评估Evans→Crimmins/Oppolzer外推能力；聚类分析诊断数据覆盖度

## [8] Feed-Forward NN for Asymmetric Negishi Reaction
- **体系**: Pd催化不对称Negishi偶联
- **任务**: 回归（ee%）
- **数据量**: **17个训练配体** + 10验证 + 3外推
- **输入表示**: **DFT-TS描述符**（Pd偏移/键角/键长/电子密度，~15个特征）
- **模型**: FFNN（2层×15节点），LOO-CV
- **关键性能**: 训练RMSE=6.9; 外推RMSE=7.73
- **启示**: DFT-TS描述符在N=17时仍有效；可引入醛醇TS关键几何描述符（B-O角/金属偏离平面度）

## [9] Predicting Enantioselective Catalysts via Tunable Fragment Descriptors
- **体系**: 手性Bronsted酸(IDP)催化
- **任务**: 回归/虚拟筛选（er预测）
- **数据量**: 数十个训练 → 虚拟筛选大量组合
- **输入表示**: **可调片段描述符**（2D指纹改进版，针对环状/多环骨架）
- **模型**: GB/RF
- **关键性能**: 最佳预测er=91.5:8.5，超越所有训练集催化剂
- **启示**: 可为Evans辅基设计定制片段描述符（噁唑烷酮/N-酰基片段），2D层面捕捉手性结构变异

## [12] Data Science for Stereoconvergent Nickel-Catalyzed Reduction
- **体系**: Ni催化烯醇甲苯磺酸酯立体汇聚还原
- **任务**: 多目标优化（产率+E/Z选择性）
- **数据量**: ~25个训练反应 → 搜索空间30,000
- **输入表示**: Sterimol参数 + 电子参数 + 反应条件
- **模型**: 统计建模 + 多目标贝叶斯优化（EDBO+）
- **关键性能**: 25个实验即收敛到~90:10 dr, >90%产率
- **启示**: 贝叶斯优化在小数据闭环验证中极高效；Sterimol描述符聚类可评估化学空间覆盖

## [13] ML for Magnesium-Catalyzed Asymmetric Reactions
- **体系**: Mg催化不对称还原/Michael加成
- **任务**: 分类（催化剂推荐排名）
- **数据量**: 文献策展（asymmetric.icho.edu.pl）
- **输入表示**: Morgan FP（**64-bit**, r=3）+ 溶剂**物化性质向量16d**
- **模型**: NN（Optuna 300步超参搜索），5折CV
- **关键性能**: ~80%准确率
- **启示**: 低维指纹(64)仍有效；**溶剂编码为物化性质向量**（替代one-hot）可直接迁移；Optuna超参搜索应引入

## [15] ML for asymmetric hydrogenation catalyst discovery（⚠️ 警示性论文）
- **体系**: Rh催化不对称烯烃加氢，192配体×5底物
- **任务**: 转化率分类 + ee回归
- **数据量**: HTE 3,552点 → 建模用960
- **输入表示**: DFT描述符(34d) vs ECFP4(512-bit) vs One-hot
- **模型**: RF（Auto-Sklearn/TPOT预筛）
- **关键性能**: 域内R²<0.2; 域外R²最高0.68(DFT) / 对不相关底物R²=-0.3
- **启示**: ⚠️ **DFT描述符不一定优于2D指纹**；域外泛化极差；AldolRxnMaster需做scaffold-split严格验证

## [20] Generality-Driven Optimization of Enantio-/Regioselective Mono-Reduction
- **体系**: COBI催化不对称1,2-二羰基单还原
- **任务**: 回归（ee%+区域选择性）
- **数据量**: 31催化剂×8底物 ≈ 248 HTE数据点
- **输入表示**: CGR（Condensed Graphs of Reaction）描述符
- **模型**: ML（可能RF/GBDT）
- **关键性能**: 工作流加速8倍；成功推荐催化剂并实验验证
- **启示**: CGR将反应编码为单一分子图，可替代Morgan+条件特征的拼接策略

## [22] ML for amidase enantioselectivity and variant design
- **体系**: 红球菌酰胺酶催化动力学拆分/去对称化
- **任务**: **分类**（ΔΔG‡阈值划分高/低选择性）
- **数据量**: 240条
- **输入表示**: clique向量(32d) + wACSF几何描述符(直方图)
- **模型**: RF（优于SVM/LR/GBDT）
- **关键性能**: F-score=0.831, AUC=0.997; 工程化变体E值提升53倍
- **启示**: **与本项目高度相似**（小数据/分类/RF）；ΔΔG‡阈值分类验证"分类优于回归"；wACSF几何描述符可补充Morgan指纹

## [23] DL for asymmetric β-C–H bond activation
- **体系**: Pd催化不对称β-C(sp3)-H官能化
- **任务**: 回归（ee%）
- **数据量**: 240条
- **输入表示**: **DFT三体复合物描述符**（金属-配体-底物TS）
- **模型**: DNN
- **关键性能**: RMSE=6.3±0.9% ee; OOB外推RMSE 5-8%
- **启示**: xTB半经验TS描述符可能是DFT和纯指纹之间的甜点（与B1-xTB对接）；特征归因方法值得引入

## [24] ML for Enantioselective C–H Activation: Generative AI to Experiment
- **体系**: 不对称β-C(sp3)-H键活化（[23]后续）
- **任务**: ee回归 + 生成式配体设计
- **数据量**: 220条反应 + 77个配体 + ~1M预训练分子
- **输入表示**: SMILES → 化学语言模型(CLM)编码
- **模型**: 迁移学习——预训练CLM(1M分子) → fine-tune；30个CLM集成预测
- **关键性能**: 实验验证"excellent agreement"
- **启示**: 预训练SMILES语言模型(MolBERT/ChemBERTa)替代或补充Morgan指纹；生成式设计可延伸到Evans辅基虚拟筛选

## [26] GNN for Data Checking of Asymmetric Catalysis Literature
- **体系**: Rh催化不对称1,4-加成
- **任务**: 回归（%top面加成百分比），用于**文献数据质量检测**
- **数据量**: 1,332条（二烯688+双膦644）
- **输入表示**: 分子图，节点特征含R/S构型
- **模型**: HCat-GNet（GNN集成，嵌套10折CV）
- **关键性能**: 人工核查范围缩减至2.2-3.5%；成功识别文献标注错误
- **启示**: **数据质量检测**——可用GNN异常检测清洗我们1654条文献数据的立体化学标注；分子图显式编码R/S构型

## [27] Homogeneous Catalyst GNN for Ligand Optimization (HCat-GNet)
- **体系**: Rh催化不对称1,4-加成 + 碘(III)CADA + 手性磷酸缩醛
- **任务**: 分类（对映面）+ 回归（ΔΔG‡）
- **数据量**: 600-1075条/反应类型
- **输入表示**: 仅SMILES（无手动特征工程）
- **模型**: GNN + GNNExplainer + SHAP
- **关键性能**: 与GB相当(p=0.825)；未知配体排序显著优于GB(p<0.01)
- **启示**: 双任务(分类+回归)框架可同时预测4-class+de%；GNNExplainer可识别关键子结构

## [28] ChIRo: 3D Chirality Representations Invariant to Bond Rotations
- **体系**: 含四面体手性中心的立体异构体
- **任务**: 对比学习 + R/S分类 + 旋光度回归
- **数据量**: 四个基准数据集
- **输入表示**: 3D构象 + 扭转角，SE(3)-不变表示
- **模型**: ChIRo（SE(3)不变+键旋转不变NN）
- **关键性能**: 手性敏感函数SOTA
- **启示**: 扭转角作为手性编码可与B1-xTB对接（xTB优化构象的扭转角特征）

## [29] ChiENN: Molecular Chirality with GNN
- **体系**: 含手性中心的化合物（MoleculeNet基准）
- **任务**: 手性敏感分子性质预测
- **数据量**: MoleculeNet标准基准
- **输入表示**: 3D分子图
- **模型**: ChiENN——模块化消息传递层，可即插即用到任意GNN
- **关键性能**: outperform当前SOTA
- **启示**: 手性感知不需要从头建模，可作为插件层；邻居空间顺序敏感机制可启发特征工程

## [30] 3DReact: Geometric Deep Learning for Chemical Reactions
- **体系**: 有机反应（活化能预测），GDB7-22-TS/Cyclo-23-TS/Proparg-21-TS
- **任务**: 回归（活化能/反应势垒）
- **数据量**: ~3K-12K
- **输入表示**: 反应物+产物3D结构 + 原子映射
- **模型**: 对称性自适应几何DL（不变/等变双通道）
- **关键性能**: MAE稳健（多种split/映射方案下）
- **启示**: 反应物+产物3D→反应表示思路可用于C1-qTS；不变/等变双通道区分手性相关/无关信息

## [32] Root-Aligned SMILES (R-SMILES) for Reaction Prediction
- **体系**: 有机合成正向/逆合成
- **任务**: 序列到序列翻译
- **数据量**: USPTO-50K / USPTO-MIT / USPTO-FULL
- **输入表示**: R-SMILES（产物-反应物原子对齐，最小化编辑距离）
- **模型**: Transformer (OpenNMT)
- **关键性能**: 正向Top-1=92.3%; 逆合成Top-1=56.3%
- **启示**: 根对齐思路可提取反应差异指纹（reaction difference FP），减少非反应位点噪声

## [33] fragSMILES: Chirality & Fragment Representation
- **体系**: 有机反应（USPTO），正/逆合成
- **任务**: 分类（Top-1产物/前体预测）
- **数据量**: 1,002,602条（手性子集8,588条）
- **输入表示**: fragSMILES（片段级手性感知文本表示）
- **模型**: Transformer (seq2seq)
- **关键性能**: 手性子集Top-1: 44.3% vs SMILES 38.8%（+5.5pp）
- **启示**: 片段级+手性感知表示提升立体化学预测；可引入BRICS片段计数作为补充特征

## [34] SIMG: Stereoelectronics-Infused Molecular Graphs
- **体系**: QM7/QM9量子化学benchmark
- **任务**: 回归（HOMO/LUMO/偶极矩等）
- **数据量**: QM9 ~134K; QM7 ~7K
- **输入表示**: SIMG——2D图+NBO轨道/孤对电子/相互作用节点; SIMG*可秒级近似
- **模型**: 双层GNN（先预测NBO再预测性质）
- **关键性能**: 接近/超过化学精度阈值
- **启示**: NBO E(2)立体电子描述符对Zimmerman-Traxler TS至关重要；SIMG*近似可降计算成本（C1阶段参考）

## [35] ChiDeK: Chiral Determinant Kernels
- **体系**: 中心手性+轴手性（ECD/旋光度）
- **任务**: R/S分类 + ECD/旋光度回归
- **数据量**: 新构建轴手性ECD/OR benchmark
- **输入表示**: Chiral Determinant Kernel（SE(3)-不变手性矩阵 + cross-attention融合）
- **模型**: GNN + cross-attention手性融合模块
- **关键性能**: 轴手性准确率提升>7%
- **启示**: **连续手性描述符**（四面体体积符号行列式值）可替代离散CIP标签；cross-attention局部→全局策略

## [37] Chirality Descriptors from SMILES Heteroencoders
- **体系**: 手性小分子（手性柱洗脱序预测）
- **任务**: R/S分类 + 洗脱序预测
- **数据量**: 3,858分子（1,929对映体对）
- **输入表示**: Heteroencoder潜空间(512d) vs **Morgan FP(r=3, 512-bit, includeChirality)**
- **模型**: RF（100棵树）
- **关键性能**: 洗脱序: **Morgan 0.822** > Transformer 0.753; R/S: **FP 0.954** > Heteroencoder 0.894
- **启示**: ⭐ **Morgan FP+includeChirality仍是强基线**，验证了我们当前方案的合理性；"Delta描述符"（FP-FP对映体）可放大手性信号

## [38] Chirality-Focused GNN for Liver Microsome Clearance
- **体系**: 人肝微粒体(HLM)药物清除率
- **任务**: 回归（清除率）
- **数据量**: 数千条HLM数据
- **输入表示**: 2D分子图（含手性信息）
- **模型**: TetraDMPNN（四面体手性感知DMPNN）/ ChIRo / RF / DNN
- **关键性能**: TetraDMPNN R²=0.639, RMSE=0.429（最优）
- **启示**: TetraDMPNN四面体手性消息传递是GNN升级首选；但中等数据下R²仅0.639，说明我们tree model的TSCV=0.795已相当不错

## [39] %VBur Index and Steric Maps（综述）
- **体系**: 过渡金属催化（NHC/膦配体空间位阻）
- **任务**: 综述——%VBur作为空间位阻描述符用于ML
- **数据量**: 综述性质
- **输入表示**: 3D计算的%VBur + steric maps
- **模型**: 综述覆盖多种ML
- **关键性能**: %VBur与化学性质相关性优于纯电子参数
- **启示**: 为Evans辅基计算%VBur可补充手性特征，对syn/anti选择性可能有直接贡献

## [40] ML-guided strategies for reaction conditions design（综述）
- **体系**: Buchwald-Hartwig/Suzuki/Heck等偶联反应
- **任务**: 反应条件预测与优化（综述）
- **数据量**: 全局库Reaxys 6500万; 局部HTE 24-5760条
- **输入表示**: ECFP/DRFP/GNN/SMILES+Transformer/CGR
- **模型**: DNN/GNN/VAE/GBM/Transformer/贝叶斯优化
- **关键性能**: 因任务而异（综述）
- **启示**: DRFP（差分反应指纹）编码底物→产物变化值得尝试；需警惕模型仅记忆高频条件组合

## [41] ML Strategies for Reaction Development: Low-Data Limit
- **体系**: 碳水化合物立体预测 / Pd偶联 / O-糖基化等
- **任务**: 低数据量反应结果预测（综述/方法论）
- **数据量**: 预训练~1M → 微调~20K → 局部数十到百条
- **输入表示**: SMILES(Transformer) / 描述符 / 特征空间(主动学习)
- **模型**: Transformer(GPT化学版) / RF迁移学习 / DL微调
- **关键性能**: 迁移学习后立体预测Top-1准确率70%（比纯目标域+40%）
- **启示**: 迁移学习（大醛醇库预训练→Evans微调）；需警惕负迁移；主动学习指导实验数据补充

## [42] Rethinking chemical research in the age of LLMs（Perspective）
- **体系**: 化学研究全流程（非特定反应）
- **任务**: LLM在化学中的五大应用范式
- **数据量**: 无特定数据集
- **输入表示**: 多模态（文本/图像/分子/光谱）
- **模型**: Coscientist / ChemCrow等LLM-agent系统
- **关键性能**: 未报告定量指标
- **启示**: 对当前项目直接启示有限；可作为Discussion前瞻性引用

## [43] Foundation Models in Chemistry（Perspective）
- **体系**: 分子性质/晶体/力场/反应/逆向设计（综述）
- **任务**: 基础模型在化学各子领域的应用
- **数据量**: PubChem 10-77M / ZINC 1B-2M / ChEMBL 456K
- **输入表示**: SMILES/SELFIES/分子图/3D/文本/混合模态
- **模型**: GraphCL/MolCLR/GROVER/ChemBERTa-2/MoLFormer等
- **关键性能**: MolCLR在MoleculeNet分类SOTA
- **启示**: 短期不需要基础模型；中长期可用ChemBERTa-2/MoLFormer做预训练embedding替代Morgan指纹

---

## 跨论文启示汇总

### 当前方案验证
- [37] 直接证明 **Morgan FP(includeChirality)+RF/Tree 是手性预测强基线**
- [38] 中等数据GNN R²仅0.639，**我们TSCV=0.795的tree model已是优秀表现**

### 短期可做（特征增强）
| 来源 | 建议 | 难度 |
|------|------|------|
| [13] | 溶剂编码改为**物化性质向量**替代one-hot | 低 |
| [13] | 引入**Optuna超参搜索**调优ExtraTrees | 低 |
| [37] | **Delta手性描述符**：FP(分子)-FP(对映体) | 中 |
| [9] | Evans辅基**定制片段描述符**（噁唑烷酮/N-酰基） | 中 |
| [33] | BRICS**片段计数特征** | 中 |
| [35] | **连续手性行列式**替代离散CIP标签 | 中 |
| [39] | Evans辅基**%VBur空间位阻描述符** | 中 |

### 中期路线（验证与策略）
| 来源 | 建议 | 难度 |
|------|------|------|
| [15] | ⚠️ **scaffold-split**严格域外验证（DFT不保证优于2D） | 中 |
| [3],[7] | **化学空间覆盖度**审计（PCA+k-means/CVT） | 中 |
| [26] | GNN异常检测做**文献数据质量清洗** | 高 |
| [41] | **迁移学习**：大醛醇库预训练→Evans微调 | 高 |

### 长期升级路径
| 来源 | 方向 |
|------|------|
| [5],[8],[23] | B1-xTB → **过渡态描述符** |
| [34] | C1阶段引入**NBO E(2)立体电子描述符** |
| [38],[29] | 数据增长后升级到**TetraDMPNN/ChiENN** |
| [24],[43] | **预训练分子语言模型**替代Morgan指纹 |
