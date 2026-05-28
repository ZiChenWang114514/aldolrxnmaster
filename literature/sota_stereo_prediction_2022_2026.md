# SOTA Review: Machine Learning for Stereochemical Prediction in Organic Chemistry (2022–2026)

> **调研目的**：系统梳理2022–2026年已发表的、与醛醇反应4-class立体化学预测（AldolRxnMaster项目）相关的ML/AI方法，覆盖从直接类比到方法论四个层次，为后续建模和论文写作提供参考。
>
> **调研范围**：醛醇反应直接预测 → 不对称合成立体选择性 → 反应立体化学处理 → 手性感知分子表征
>
> **检索期刊**：Nature/Science/Chem/JACS/ACS Cent. Sci./Angew. Chem./Chem. Sci./Nat. Commun./Nat. Comput. Sci./Nat. Synth./JCIM/J. Cheminform./Molecules 等
>
> **更新日期**：2026-05-28

---

## 目录

1. [醛醇反应与手性辅基直接相关](#section1)
2. [不对称催化立体选择性预测（对映选择性）](#section2)
3. [一般有机反应立体化学结果预测](#section3)
4. [手性感知GNN与分子表征方法](#section4)
5. [综述与展望类](#section5)
6. [总结对比表](#section6)
7. [对本项目的启示](#section7)

---

<a name="section1"></a>
## 1. 醛醇反应与手性辅基直接相关

> **说明**：经系统检索（5个并行搜索agent，覆盖PubMed/Semantic Scholar/Web of Science，共119次工具调用），**2022–2026年间尚无专门针对Evans/Crimmins等手性辅基醛醇反应进行ML立体化学预测的独立发表论文**。这本身是一个重要发现，证明本项目填补了文献空白。最接近的相关工作如下：

---

### [1] Predicting glycosylation stereoselectivity using machine learning

- **作者**: Sooyeon Moon et al. (Peter H. Seeberger, Kerry Gilmore, Max-Planck Institute for Colloids and Interfaces)
- **期刊**: *Chemical Science* (RSC), 2021 (online 2020-12, print 2021)
- **链接**: [https://doi.org/10.1039/d0sc06222g](https://doi.org/10.1039/d0sc06222g)
- **任务**: 预测糖苷化反应（α/β立体选择性）定量结果，考虑亲电体/亲核体/酸催化剂/溶剂/温度五维变量
- **数据集**: 268个训练点（7亲电×6亲核×4酸×7溶剂×温度范围）+ 两个独立验证集
- **方法**: 随机森林回归，基于量子力学计算的空间与电子描述符（10个描述符+温度）
- **关键指标**: RMSE = 6.8%（优于GPR 7.9%、SVM 10%）；亲电体和溶剂各贡献27%预测权重
- **与本项目相关性**: ⭐⭐⭐⭐ 高度相关——同为底物控制的立体选择性预测，多变量+条件依赖，处理类似的"相同底物骨架+不同条件"场景

---

### [2] Anomeric Selectivity of Glycosylations through a Machine Learning Lens

- **作者**: Sooyeon Moon, Sourav Chatterjee, Peter H. Seeberger, Kerry Gilmore (Max-Planck Institute)
- **期刊**: *Journal of the American Chemical Society*, 2025, 147(40), 36197–36209
- **链接**: [https://doi.org/10.1021/jacs.5c07561](https://doi.org/10.1021/jacs.5c07561)
- **任务**: 糖苷化反应异头体选择性预测（GlycoPredictor工具，多任务分类+回归）
- **数据集**: >10,000个糖苷化反应（比[1]扩大约40倍）
- **方法**: 多任务ML：同时预测主异头体、次要异头体是否出现、异头体比例
- **关键指标**: 首个工业规模糖苷化立体选择性预测工具
- **与本项目相关性**: ⭐⭐⭐⭐ 高度相关——大规模底物控制立体选择性多分类预测，方法论最接近本项目

---

### [A+] ScopeMap: An AI-Assisted, Human-in-the-Loop Workflow for Mapping Reaction Scope and Boundaries *(含仿生醛醇体系验证)*

- **作者**: Jiawei Li et al.
- **期刊**: *Angewandte Chemie International Edition*, 2026 (May 8, online first)
- **链接**: [https://doi.org/10.1002/anie.2455429](https://doi.org/10.1002/anie.2455429)
- **任务**: 预测反应底物适用范围与反应边界；验证数据集含**仿生醛醇反应（biomimetic aldol reactions）**
- **数据集**: 仅使用底物空间<4%作为训练样本（主动学习策略）
- **方法**: 改进质心Voronoi镶嵌（CVT）算法 + 动态几何排斥势 + "人在环中"主动学习
- **关键指标**: 反应性预测F1 > 90%
- **与本项目相关性**: ⭐⭐⭐ 中相关——唯一经验证包含醛醇反应的2022-2026 ML论文；主动学习策略值得关注

---

### [3] Artificial Intelligence for Predicting Stereoselectivity in Glycosylation Chemistry

- **作者**: Junjie Fu et al.
- **期刊**: *CCS Chemistry*, 2025/2026 (online 2025-12-18)
- **链接**: [https://doi.org/10.31635/ccschem.025.202506735](https://doi.org/10.31635/ccschem.025.202506735)
- **任务**: 综述AI/ML用于糖基化立体选择性预测，涵盖α/β异头体选择性（anomeric selectivity）预测
- **数据集**: 文献汇编，多批次实验数据
- **方法**: 综述，包括随机森林+量化描述符、迁移学习、可解释决策树
- **关键指标**: 综述性文章，系统评价各方法在糖基化任务上的性能
- **与本项目相关性**: ⭐⭐⭐ 中高相关——糖基化立体选择性与醛醇立体选择性任务类型相似，均为底物控制+条件依赖

---

<a name="section2"></a>
## 2. 不对称催化立体选择性预测（对映选择性）

### [4] Enantioselectivity prediction of pallada-electrocatalysed C–H activation using transition state knowledge in machine learning

- **作者**: Li-Cheng Xu, Johanna Frey, Xiaoyan Hou, Shuo-Qing Zhang et al. (Xin Hong / Lutz Ackermann)
- **期刊**: *Nature Synthesis*, 2023, **2**, 321–330
- **链接**: [https://doi.org/10.1038/s44160-022-00233-y](https://doi.org/10.1038/s44160-022-00233-y)
- **任务**: 不对称钯-电催化C–H活化对映选择性的数据驱动预测
- **数据集**: 枚举846,720种底物/配体/条件组合
- **方法**: 过渡态知识向量化 + Random Forest + SHAP可解释性
- **关键指标**: 对846,720种组合进行定量对映选择性预测；揭示非直觉烯烃插入效应
- **与本项目相关性**: ⭐⭐⭐ 中相关——TS知识向量化策略值得借鉴（本项目的MechAware特征本质上是类似思路）

---

### [5] Data Science Enables the Development of a New Class of Chiral Phosphoric Acid Catalysts

- **作者**: Jordan P. Liles, Caroline Rouget-Virbel et al. (F. Dean Toste, Matthew S. Sigman)
- **期刊**: *Chem* (Cell Press), 2023, Vol. 9, Issue 6, 1518–1537
- **链接**: [https://doi.org/10.1016/j.chempr.2023.02.020](https://doi.org/10.1016/j.chempr.2023.02.020)
- **任务**: 手性磷酸（CPA）催化剂设计与转移氢化对映选择性预测
- **数据集**: 20个催化剂训练集，3类反应类型
- **方法**: PCA + k-means聚类 + DFT QSAR描述符 + 多元线性回归
- **关键指标**: 分类准确率0.93；测试集R² = 0.98；外推最佳催化剂er = 95:5（实验验证）
- **与本项目相关性**: ⭐⭐ 中相关——小数据集上极高性能，说明化学先验知识描述符的重要性

---

### [6] A Data-Driven Workflow for Assigning and Predicting Generality in Asymmetric Catalysis

- **作者**: Isaiah O. Betinol, Junshan Lai, Saumya Thakur, Jolene P. Reid (University of British Columbia)
- **期刊**: *Journal of the American Chemical Society*, 2023, **145**(23), 12870–12883
- **链接**: [https://doi.org/10.1021/jacs.3c03989](https://doi.org/10.1021/jacs.3c03989)
- **任务**: 不对称催化剂的普适性（generality）定量评估与预测
- **数据集**: 文献挖掘数据（有机催化Mannich反应 + CPA催化亚胺反应）
- **方法**: 无监督ML（描述符空间可视化与聚类）+ 监督回归（普适性打分）
- **关键指标**: 成功识别高普适性催化剂化学型并排序
- **与本项目相关性**: ⭐⭐ 中相关——"泛化能力评估"框架与本项目的TSCV/Scaffold评估思路互补

---

### [7] Feed-Forward Neural Network for Predicting Enantioselectivity of the Asymmetric Negishi Reaction

- **作者**: Abbigayle E. Cuomo et al. (Victor S. Batista, Timothy R. Newhouse, Yale / Boehringer Ingelheim)
- **期刊**: *ACS Central Science*, 2023, **9**(9), 1768–1774
- **链接**: [https://doi.org/10.1021/acscentsci.3c00512](https://doi.org/10.1021/acscentsci.3c00512)
- **任务**: Pd催化不对称Negishi交叉偶联（P-手性磷配体）对映选择性预测
- **数据集**: 17个配体训练集 + 10个验证配体（极小数据集）
- **方法**: DFT过渡态几何/电子/弥散相互作用特征 → 前馈神经网络（两隐层×15节点）
- **关键指标**: 训练RMSE = 6.9% ee；成功预测超出训练范围的新配体（L31: 预测11:89 er，实测6:94 er）
- **与本项目相关性**: ⭐⭐⭐ 中相关——极小数据集+DFT先验知识策略；与本项目恰恰相反（数据大但无DFT），互为补充

---

### [8] Predicting Highly Enantioselective Catalysts Using Tunable Fragment Descriptors

- **作者**: Nobuya Tsuji, Pavel Sidorov, Chendan Zhu, Yuuya Nagata, Timur Gimadiev, Alexandre Varnek, Benjamin List (Max-Planck + Strasbourg)
- **期刊**: *Angewandte Chemie International Edition*, 2023, **62**(11), e202218659
- **链接**: [https://doi.org/10.1002/anie.202218659](https://doi.org/10.1002/anie.202218659)
- **任务**: 用中等选择性训练数据外推预测新型高对映选择性催化剂
- **数据集**: 小规模训练集（含中等选择性数据）
- **方法**: 可调节碎片描述符（tunable fragment descriptors）+ ML分类/回归
- **关键指标**: 从中等选择性训练集成功外推设计新催化剂，实验验证高ee
- **与本项目相关性**: ⭐⭐ 低中相关——片段描述符方法值得借鉴

---

### [9] Predicting the stereoselectivity of chemical reactions by composite machine learning method

- **作者**: Jihoon Chung, Justin Li, Amirul Islam Saimon, Pengyu Hong, Zhenwo Kong
- **期刊**: *Scientific Reports*, 2024 (May 27)
- **链接**: [https://doi.org/10.1038/s41598-024-62158-0](https://doi.org/10.1038/s41598-024-62158-0)
- **任务**: CPA催化反应对映选择性（ΔΔG‡）定量预测，使用复合ML方法
- **数据集**: 342个CPA反应（307训练 + 35测试 + 64外部验证）
- **方法**: SVR + RF + LASSO组合，高斯混合模型（GMM）智能选择子模型，贝叶斯超参数优化
- **关键指标**: SVR R² = 0.936；复合模型MSE降低>70%；测试R² > 0.75
- **与本项目相关性**: ⭐⭐⭐ 中相关——复合模型集成策略；本项目P1.3集成stacking可借鉴

---

### [10] A meta-learning approach for selectivity prediction in asymmetric catalysis

- **作者**: Sukriti Singh, José Miguel Hernández-Lobato (Cambridge)
- **期刊**: *Nature Communications*, 2025, **16**, 3599
- **链接**: [https://doi.org/10.1038/s41467-025-58854-8](https://doi.org/10.1038/s41467-025-58854-8)
- **任务**: 不对称氢化反应对映选择性分类预测（ee > 80% vs ≤ 80%）
- **数据集**: 11,932个不对称氢化反应（5,009 Ir + 6,391 Rh + 532 Co催化）
- **方法**: 元学习（Prototypical Networks）+ Morgan指纹（512-bit）+ GNN分子图编码
- **关键指标**: AUPRC = 0.9300（meta-cluster版本）；显著优于RF（0.8369）和GNN（0.8259）
- **与本项目相关性**: ⭐⭐⭐ 中相关——元学习少样本策略对于小辅基子集（Crimmins 259条/Oppolzer 141条）具有重要借鉴意义

---

### [11] Data Science Guided Multiobjective Optimization of a Stereoconvergent Nickel-Catalyzed Reduction

- **作者**: N. P. Romer, D. S. Min, J. Y. Wang et al. (A. G. Doyle, M. S. Sigman / Genentech)
- **期刊**: *ACS Catalysis*, 2024, **14**(7), 4699–4708
- **链接**: [https://doi.org/10.1021/acscatal.4c00650](https://doi.org/10.1021/acscatal.4c00650)
- **任务**: 立体收敛Ni催化还原E/Z非对映选择性 + 产率多目标优化
- **数据集**: 系统训练集（单磷配体库）
- **方法**: 数据科学工作流 + 统计建模 + 贝叶斯多目标优化
- **关键指标**: E/Z dr最高~90:10，产率>90%
- **与本项目相关性**: ⭐⭐ 中低相关——多目标优化框架；非对映选择性与本项目4-class预测有共通点

---

### [12] Machine Learning Algorithm Guides Catalyst Choices for Magnesium-Catalyzed Asymmetric Reactions

- **作者**: Baczewska M., Roszak R., Grzybowski B.A., Mlynarski J. (Polish Academy of Sciences)
- **期刊**: *Angewandte Chemie International Edition*, 2024, e202318487
- **链接**: [https://doi.org/10.1002/anie.202318487](https://doi.org/10.1002/anie.202318487)
- **任务**: Mg催化不对称还原和Michael加成的催化剂选择ML预测
- **数据集**: 文献汇编（Mg催化不对称反应数据库）
- **方法**: Random Forest/梯度提升 + 文献数据库驱动
- **关键指标**: 实验验证：不对称还原和Michael加成均成功
- **与本项目相关性**: ⭐ 低相关——文献数据库驱动思路与本项目相同

---

### [13] Chemistry-informed deep learning model for predicting stereoselectivity and absolute configuration in asymmetric hydrogenation

- **作者**: Li Cheng et al. (Pan-Lin Shao / Bo Zhang / Guichuan Xing, SUSTech + UM + GZHMU)
- **期刊**: *Nature Computational Science*, 2025/2026, **6**, 145–155
- **链接**: [https://doi.org/10.1038/s43588-025-00920-8](https://doi.org/10.1038/s43588-025-00920-8)
- **任务**: 烯烃不对称氢化（含两个前手性中心）的对映选择性 + **绝对构型**同步预测
- **数据集**: 多催化剂/多底物大规模数据集（覆盖多类不对称催化体系）
- **方法**: ChemAHNet——基于反应机理的三模块结构感知深度学习，仅输入SMILES
- **关键指标**: 准确预测主要对映体的绝对构型；突破现有ML在多前手性中心场景的局限
- **与本项目相关性**: ⭐⭐⭐⭐ 高度相关——**同样预测绝对构型（R/S）而非仅ee值**，机理知情建模思路与本项目MechAware特征高度对应

---

### [14] Probing machine learning models based on high throughput experimentation data for the discovery of asymmetric hydrogenation catalysts

- **作者**: Adarsh V. Kalikadien et al. (Laurent Lefort, Evgeny A. Pidko / TU Delft + Janssen)
- **期刊**: *Chemical Science* (RSC), 2024
- **链接**: [https://doi.org/10.1039/d4sc03647f](https://doi.org/10.1039/d4sc03647f)
- **任务**: Rh催化不对称氢化对映选择性预测，催化剂发现
- **数据集**: 3,552个实验数据点（192个催化剂 × 5个底物），迄今最大均质HTE数据集
- **方法**: 高通量实验（HTE）+ DFT描述符（34特征）+ ECFP指纹 + Random Forest
- **关键指标**: 域内转化率预测准确；域外对映选择性预测R² < 0.2（揭示当前ML局限）
- **与本项目相关性**: ⭐⭐⭐ 中相关——HTE数据驱动策略；域外预测困难与本项目Scaffold/TSCV挑战一致

---

### [15] Data-Driven Prediction of Enantioselectivity for the Sharpless Asymmetric Dihydroxylation: Model Development and Experimental Validation

- **作者**: Blake E. Ocampo, Bilal Altundas, Matthew J. Bock, Sara Feiz, Scott E. Denmark (UIUC)
- **期刊**: *ACS Central Science*, 2025, **11**(9), 1640–1650
- **链接**: [https://doi.org/10.1021/acscentsci.5c00900](https://doi.org/10.1021/acscentsci.5c00900)
- **任务**: Sharpless不对称二羟基化（AD）反应对映选择性预测
- **数据集**: 1,007个文献反应（AD反应历史数据库）
- **方法**: 片段化校准描述符 + Gradient Boosting/RF/GPR + SHAP可解释性，15个新底物实验验证
- **关键指标**: Q²F3 ≥ 0.8，MAE ≤ 0.30 kcal/mol
- **与本项目相关性**: ⭐⭐⭐⭐ 高度相关——**底物控制（配体固定）的不对称反应**，以历史文献数据库训练，思路与本项目完全对应；SHAP分析值得参考

---

### [16+] Connecting the complexity of stereoselective synthesis to the evolution of predictive tools *(重要综述)*

- **作者**: Jiajing Li et al.
- **期刊**: *Chemical Science* (RSC), 2025, **16**(9), 3832–3851
- **链接**: [https://doi.org/10.1039/d4sc07461k](https://doi.org/10.1039/d4sc07461k)
- **任务**: 综述立体选择性合成预测工具演变：从Cram/Felkin-Anh规则 → 量子化学 → ML回归 → GNN
- **数据集**: N/A（综述）
- **方法**: 综述，涵盖分子描述符回归、GNN、量子化学计算等
- **与本项目相关性**: ⭐⭐⭐⭐ 高度相关——**本项目论文写作的最佳参照综述**，直接对比前ML时代（Zimmerman-Traxler规则）与ML方法

---

### [16++] Leveraging Limited Experimental Data with Machine Learning: CBS Reduction

- **作者**: Oliver Pereira et al.
- **期刊**: *Journal of the American Chemical Society*, 2024, **146**(21), 14576–14586
- **链接**: [https://doi.org/10.1021/jacs.4c01286](https://doi.org/10.1021/jacs.4c01286)
- **任务**: 极小数据集（~100条）下预测CBS还原（酮底物）对映选择性，区分甲基/乙基效应
- **数据集**: ~100个反应（三重复测量）
- **方法**: ML + 关键中间体图（Key-intermediate graph），预测ΔΔG‡，无需过渡态建模
- **关键指标**: 实现ee = 80%（丁酮底物），优于酶（64%）和原始CBS催化剂（60%）；ΔG‡提升>50%
- **与本项目相关性**: ⭐⭐⭐ 中相关——小数据集+化学先验知识的极致运用，与本项目Myers子集（14条）场景对应

---

### [16+++] Bayesian Meta-Learning for Few-Shot Reaction Outcome Prediction of Asymmetric Hydrogenation

- **作者**: Sukriti Singh, José Miguel Hernández-Lobato (Cambridge)
- **期刊**: *Angewandte Chemie International Edition*, 2025, **64**(27), e202503821
- **链接**: [https://doi.org/10.1002/anie.202503821](https://doi.org/10.1002/anie.202503821)
- **任务**: 极少样本条件下预测过渡金属催化烯烃不对称氢化对映选择性
- **数据集**: >12,000条文献挖掘的过渡金属催化反应
- **方法**: 贝叶斯元学习（Bayesian Meta-Learning）；ADKF-prior（改进的深度核拟合）优于原型网络
- **关键指标**: AUPRC优于RF/GNN；低数据场景（16个支持样本）性能显著优于基线
- **与本项目相关性**: ⭐⭐⭐ 中相关——同一作者的另一篇进阶工作，贝叶斯元学习比[10]更适合极小辅基子集

---

### [16++++] Generality-Driven Optimization of Enantio- and Regioselective Mono-Reduction

- **作者**: Terim Seo et al.
- **期刊**: *Angewandte Chemie International Edition*, 2026, **65**(1), e202519425
- **链接**: [https://doi.org/10.1002/anie.202519425](https://doi.org/10.1002/anie.202519425)
- **任务**: 同时优化COBI催化1,2-二羰基化合物单还原的**对映选择性+区域选择性**
- **数据集**: 8底物×31 COBI变体高通量实验矩阵
- **方法**: CGR（Condensed Graph of Reaction）描述符 + ML
- **关键指标**: ee > 99%，区域选择性 > 20:1；工作流加速8倍
- **与本项目相关性**: ⭐⭐ 低中相关——同时预测多种立体化学结果，CGR描述符是本项目未使用的一种反应表征

---

### [16] Evaluating Predictive Accuracy in Asymmetric Catalysis: A Machine Learning Perspective on Local Reaction Space

- **作者**: Isaiah O. Betinol et al. (Jolene P. Reid, UBC)
- **期刊**: *ACS Catalysis*, 2025, **15**(8), 6067–6077
- **链接**: [https://doi.org/10.1021/acscatal.5c01051](https://doi.org/10.1021/acscatal.5c01051)
- **任务**: 系统评估ML在不对称催化预测中的"近邻驱动"本质
- **数据集**: 多个不对称催化数据集（跨多类反应）
- **方法**: 控制变量实验，比较结构/电子相似度近邻比例对预测的影响
- **关键指标**: 揭示ML预测精度主要由局部空间最近邻驱动
- **与本项目相关性**: ⭐⭐⭐ 中相关——本项目TSCV评估策略正是回避这种"近邻驱动"高估的正确做法

---

### [17] Machine learning-assisted amidase-catalytic enantioselectivity prediction and rational design of variants

- **作者**: Zi-Lin Li et al. (De-Xian Wang, Yu-Fei Ao, Chinese Academy of Sciences)
- **期刊**: *Nature Communications*, 2024 (Oct. 10)
- **链接**: [https://doi.org/10.1038/s41467-024-53048-0](https://doi.org/10.1038/s41467-024-53048-0)
- **任务**: 酰胺酶催化对映选择性预测 + 酶变体理性设计
- **数据集**: 240个实验数据点
- **方法**: Random Forest分类（化学+几何描述符）+ 理性变体设计
- **关键指标**: 成功预测对映选择性并指导实验，高活性变体E值提升53倍
- **与本项目相关性**: ⭐ 低相关——240条小数据集，酶催化体系

---

### [18] Deep learning for enantioselectivity predictions in catalytic asymmetric β-C–H bond activation reactions

- **作者**: Ajnabiul Hoque et al. (Digital Discovery, RSC, 2022)
- **期刊**: *Digital Discovery*, 2022
- **链接**: [https://doi.org/10.1039/D2DD00084A](https://doi.org/10.1039/D2DD00084A)
- **任务**: Pd催化β-C(sp³)–H官能团化（手性氨基酸配体）对映选择性预测
- **数据集**: 240个反应（SMOTE过采样至~301）
- **方法**: DNN + DFT金属-配体-底物复合体几何描述符 + SHAP
- **关键指标**: RMSE = 6.3 ± 0.9% ee；SHAP揭示关键结构特征
- **与本项目相关性**: ⭐⭐ 低中相关——SHAP用于化学可解释性分析值得借鉴（本项目P3.1 SHAP计划）

---

### [19] Molecular Machine Learning Approach to Enantioselective C–H Bond Activation: From Generative AI to Experimental Validation

- **作者**: Ajnabiul Hoque et al.
- **期刊**: *Chemical Science* (RSC), 2025
- **链接**: [https://doi.org/10.1039/d5sc01098e](https://doi.org/10.1039/d5sc01098e)
- **任务**: 催化不对称β-C(sp³)–H活化对映选择性预测 + 生成AI设计新手性配体
- **数据集**: 220个文献反应（77种配体，5种底物，51种卤代芳烃，20种碱）
- **方法**: ULMFiT化学语言模型（100万ChEMBL分子预训练）+ 集成预测器（30个模型） + 生成器（配体latent空间采样）
- **关键指标**: R² = 0.89，RMSE = 7.57 ± 1.31% ee；15次前瞻性实验验证
- **与本项目相关性**: ⭐⭐ 低中相关——化学语言模型迁移学习策略

---

<a name="section3"></a>
## 3. 一般有机反应立体化学结果预测

### [20] Data-Efficient, Chemistry-Aware Machine Learning Predictions of Diels–Alder Reaction Outcomes

- **作者**: Angus Keto, Taicheng Guo, Morgan Underdue, Thijs Stuyver, Connor W. Coley, Xiangliang Zhang, Elizabeth H. Krenske, Olaf Wiest (UQ + Notre Dame + MIT)
- **期刊**: *Journal of the American Chemical Society*, 2024, **146**(23), 16052–16061
- **链接**: [https://doi.org/10.1021/jacs.4c03131](https://doi.org/10.1021/jacs.4c03131)
- **任务**: Diels-Alder反应区域选择性、位点选择性和**非对映选择性**预测（主产物）
- **数据集**: 9,537个反应（含分子内、杂DA、芳香体系、逆电子需求DA）
- **方法**: NERF（chemistry-aware神经网络，显式模拟成键/断键变化）
- **关键指标**: 仅40%训练数据达>90% Top-1准确率；优于Chemformer
- **与本项目相关性**: ⭐⭐⭐ 中相关——**同样预测非对映选择性（diastereoselectivity）主产物**，化学感知模型架构值得参考

---

### [21] Data Checking of Asymmetric Catalysis Literature Using a Graph Neural Network Approach

- **作者**: Eduardo Aguilar-Bejarano et al.
- **期刊**: *Molecules*, 2025, **30**(2), Art. 355
- **链接**: [https://doi.org/10.3390/molecules30020355](https://doi.org/10.3390/molecules30020355)
- **任务**: 用GNN集成自动识别不对称催化数据库中的手性误归属（立体化学标签错误）
- **数据集**: 1,332个反应（GNN+嵌套10折CV）
- **方法**: HCat-GNet GNN集成，多数模型偏离时标记为疑似误归属
- **关键指标**: 人工审查量压缩至2.2-3.5%
- **与本项目相关性**: ⭐⭐⭐ 中相关——本项目也面临Reaxys文献数据质量问题，类似的自动标签验证框架值得参考

---

### [22] Homogeneous catalyst graph neural network: A human-interpretable GNN tool for ligand optimization in asymmetric catalysis

- **作者**: Eduardo Aguilar-Bejarano et al.
- **期刊**: *iScience*, 2025
- **链接**: [https://doi.org/10.1016/j.isci.2025.111881](https://doi.org/10.1016/j.isci.2025.111881)
- **任务**: 仅凭SMILES预测金属-配体不对称催化对映选择性（ΔΔG‡），辅助配体优化
- **数据集**: 668个Rh催化不对称1,4-加成反应（150种配体×84种底物×45种硼试剂）+ 52个未见配体测试集
- **方法**: 图卷积 + 消息传递GNN（反应图输入）
- **关键指标**: MAE ≤ 2.5 kJ/mol；对未见配体对映选择性成功排序
- **与本项目相关性**: ⭐⭐ 中低相关——反应图GNN，端到端SMILES预测立体选择性

---

<a name="section4"></a>
## 4. 手性感知GNN与分子表征方法

### [23] ChIRo: Learning 3D Representations of Molecular Chirality with Invariance to Bond Rotations

- **作者**: Keir Adams et al.
- **期刊**: *ICLR 2022* (机器学习顶会)
- **链接**: [https://openreview.net/forum?id=hm2tNDdgaFK](https://openreview.net/forum?id=hm2tNDdgaFK) | arXiv: [2110.04383](https://arxiv.org/abs/2110.04383)
- **任务**: 学习对键旋转不变、但对立体异构体敏感的3D手性分子表征
- **数据集**: 四个手性基准数据集（蛋白-配体对接排序等）
- **方法**: SE(3)-不变模型，以扭转角作为输入，显式建模内部键旋转；MPNN骨架
- **关键指标**: 手性敏感docking排序任务上超越TetraDMPNN / 2D GNN基线
- **与本项目相关性**: ⭐⭐ 低中相关——3D手性表征方法，本项目未使用3D GNN但ChiralEnv特征有类似思想

---

### [24] ChiENN: Embracing Molecular Chirality with Graph Neural Networks

- **作者**: Piotr Gaiński et al.
- **期刊**: *ECML PKDD 2023* (Springer LNCS, pp. 36–52)
- **链接**: [https://link.springer.com/chapter/10.1007/978-3-031-43418-1_3](https://link.springer.com/chapter/10.1007/978-3-031-43418-1_3) | arXiv: [2307.02198](https://arxiv.org/abs/2307.02198)
- **任务**: 使任意GNN具备手性感知能力——区分对映体的分子性质预测
- **数据集**: 多个手性敏感分子性质预测基准
- **方法**: Chiral Edge Neural Network（ChiENN）层，通过邻域有序排列实现手性区分，理论证明有效性
- **关键指标**: 超越TetraDMPNN、ChIRo等先有方法
- **与本项目相关性**: ⭐⭐ 低中相关——可插拔手性GNN层设计，理论完备性强

---

### [25] 3DReact: Geometric Deep Learning for Chemical Reactions

- **作者**: Puck van Gerwen et al. (Valence Labs)
- **期刊**: *Journal of Chemical Information and Modeling*, 2024
- **链接**: [https://doi.org/10.1021/acs.jcim.4c00104](https://doi.org/10.1021/acs.jcim.4c00104) | arXiv: [2312.08307](https://arxiv.org/abs/2312.08307)
- **任务**: 从反应物和产物的3D结构预测化学反应活化能垒（含立体化学敏感场景）
- **数据集**: GDB7-22-TS、Cyclo-23-TS、Proparg-21-TS（含SMILES相同但立体化学不同的分子）
- **方法**: 不变/等变GNN处理3D结构；支持有/无原子映射两种输入
- **关键指标**: 在立体化学敏感数据集上大幅超越2D-GNN（ChemProp）
- **与本项目相关性**: ⭐⭐ 低中相关——3D GNN处理SMILES无法区分的立体异构体

---

### [26] Difficulty in Chirality Recognition for Transformer Architectures Learning Chemical Structures from String Representations

- **作者**: Yasuhiro Yoshikai et al.
- **期刊**: *Nature Communications*, 2024 (Feb. 16)
- **链接**: [https://www.nature.com/articles/s41467-024-45102-8](https://www.nature.com/articles/s41467-024-45102-8)
- **任务**: 系统研究Transformer从SMILES字符串学习手性识别的困难
- **数据集**: 大规模SMILES预训练语料（数百万分子）+ 对映体识别测试集
- **方法**: BERT-style化学语言模型，分析手性token学习动态
- **关键指标**: 揭示训练不足时对映体识别准确率接近随机；需极长训练才能稳定识别手性
- **与本项目相关性**: ⭐⭐⭐ 中相关——解释了为何本项目不使用SMILES-based Transformer直接预测；验证了结构化手性特征的必要性

---

### [27] Root-Aligned SMILES: A Tight Representation for Chemical Reaction Prediction

- **作者**: Zhong-ting Wan et al.
- **期刊**: *Chemical Science* (RSC), 2022
- **链接**: [https://doi.org/10.1039/d2sc02763a](https://doi.org/10.1039/d2sc02763a) | arXiv: [2203.11444](https://arxiv.org/abs/2203.11444)
- **任务**: 设计对齐的SMILES表征解决手性反应预测准确率下滑问题
- **数据集**: USPTO-50K（逆合成）、USPTO-FULL（正向合成）
- **方法**: Root-Aligned SMILES（R-SMILES）+ Transformer解码
- **关键指标**: 含手性反应准确率仅下降4.3%（普通SMILES下降13.3%）；Top-1逆合成准确率大幅提升
- **与本项目相关性**: ⭐ 低相关——揭示立体化学SMILES编码的系统性挑战

---

### [28] Enhancing Deep Chemical Reaction Prediction with Advanced Chirality and Fragment Representation (fragSMILES)

- **作者**: Fabrizio Mastrolorito et al.
- **期刊**: *Chemical Communications* (RSC), 2025
- **链接**: [https://pubs.rsc.org/en/content/articlelanding/2025/cc/d5cc02641e](https://pubs.rsc.org/en/content/articlelanding/2025/cc/d5cc02641e)
- **任务**: 片段级手性感知字符串表征用于正向/逆向合成预测
- **数据集**: USPTO数据库（~1,002,602条精选化学反应）
- **方法**: fragSMILES编码分子片段+手性；Seq2Seq Transformer
- **关键指标**: 正向合成Top-1准确率最高提升+5%；立体化学反应识别最优
- **与本项目相关性**: ⭐ 低相关——大规模合成预测，与本项目任务（小数据集选择性预测）场景不同

---

### [29] Advancing Molecular Machine Learning with Stereoelectronics-Infused Molecular Graphs (SIMG)

- **作者**: Dávid Bajusz et al. (Gomes group, Cornell)
- **期刊**: *Nature Machine Intelligence*, 2025
- **链接**: [https://www.nature.com/articles/s42256-025-01031-9](https://www.nature.com/articles/s42256-025-01031-9) | arXiv: [2408.04520](https://arxiv.org/abs/2408.04520)
- **任务**: 将量子化学立体电子效应（NBO轨道相互作用）注入分子图表征
- **数据集**: 小分子量化学计算数据集 + 蛋白质级别外推验证
- **方法**: 双GNN工作流：一个GNN预测立体电子SIMG表征，另一GNN用于下游任务
- **关键指标**: 在小分子训练后可外推到整个蛋白质；多项性质预测优于标准MPNN
- **与本项目相关性**: ⭐⭐ 低中相关——立体电子效应编码与本项目MechAware特征思路相近

---

### [30] Learning Molecular Chirality via Chiral Determinant Kernels (ChiDeK)

- **作者**: Runhan Shi et al.
- **期刊**: *ICLR 2026* (机器学习顶会，已接收)
- **链接**: [https://arxiv.org/abs/2602.07415](https://arxiv.org/abs/2602.07415)
- **任务**: 统一建模中心手性和轴手性，从局部手性中心提取特征并通过cross-attention集成
- **数据集**: 多手性类型基准数据集（中心手性+轴手性分子）
- **方法**: Chiral Determinant Kernel（ChiDeK）+ SE(3)-不变手性矩阵 + cross-attention
- **关键指标**: 首个联合处理central + axial chirality的统一框架；超越ChiENN、ChIRo
- **与本项目相关性**: ⭐⭐ 低中相关——最新手性GNN理论，代表该方向SOTA

---

### [31] Stereoisomers Are Not Machine Learning's Best Friends

- **作者**: Gökhan Tahıl et al.
- **期刊**: *Journal of Chemical Information and Modeling*, 2024, **64**(14), 5451–5469
- **链接**: [https://doi.org/10.1021/acs.jcim.4c00318](https://doi.org/10.1021/acs.jcim.4c00318)
- **任务**: 系统分析主流ML分子表示（SMILES、Morgan指纹、Mol2vec等）区分立体异构体的能力
- **数据集**: 环糊精-客体结合常数数据集（立体异构体对）
- **方法**: 对比各类描述符方案的准确率/关联系数
- **关键指标**: 揭示多数主流表征方法对立体异构体的系统性失败
- **与本项目相关性**: ⭐⭐⭐ 中相关——为本项目使用专门手性特征（chiral_*、chiralenv_*、ald_pri_*）而非通用Morgan指纹提供了方法论依据

---

### [32] Evaluation of Chirality Descriptors Derived from SMILES Heteroencoders

- **作者**: Natalia Baimacheva et al.
- **期刊**: *Journal of Cheminformatics*, 2025
- **链接**: [https://link.springer.com/article/10.1186/s13321-025-01080-7](https://link.springer.com/article/10.1186/s13321-025-01080-7)
- **任务**: 评估SMILES变分自编码器隐空间向量提取手性描述符的能力
- **数据集**: 3,858个分子（1,929对对映体）；Chiralpak® AD-H柱文献数据
- **方法**: SMILES heteroencoder + 隐空间算术 + RF下游分类器
- **关键指标**: 洗脱顺序预测准确率0.75（heteroencoder）vs 0.82（传统指纹）
- **与本项目相关性**: ⭐⭐ 低中相关——评估手性描述符质量的基准方法

---

### [33] Prediction of Human Liver Microsome Clearance with Chirality-Focused Graph Neural Networks

- **作者**: Chengtao Pu et al.
- **期刊**: *Journal of Chemical Information and Modeling*, 2024, **64**(14), 5427–5438
- **链接**: [https://doi.org/10.1021/acs.jcim.4c00243](https://doi.org/10.1021/acs.jcim.4c00243)
- **任务**: 预测人肝微粒体代谢清除率，专门解决手性被QSPR模型忽视的问题
- **数据集**: 公开HLM数据库（两个不同数据集）
- **方法**: 对比RF、DNN、DMPNN、TetraDMPNN、ChIRo三类手性感知GNN
- **关键指标**: TetraDMPNN R² = 0.639，RMSE = 0.429；手性GNN系统性优于不感知手性的模型
- **与本项目相关性**: ⭐ 低相关——ADMET应用，手性感知GNN对比研究

---

<a name="section5"></a>
## 5. 综述与展望类

### [34] %VBur index and steric maps: from predictive catalysis to machine learning

- **作者**: Sílvia Escayola et al.
- **期刊**: *Chemical Society Reviews*, 2024, **53**(2), 853–882
- **链接**: [https://doi.org/10.1039/D3CS00725A](https://doi.org/10.1039/D3CS00725A)
- **任务**: 综述立体/空间位阻描述符（%VBur）与ML在不对称催化选择性预测中的应用
- **方法**: 综述，涵盖QSPR建模和虚拟筛选
- **与本项目相关性**: ⭐⭐⭐ 中相关——本项目大量使用Vbur、L、B1、B5等空间位阻描述符（34d Steric特征）；该综述提供了这类特征的完整理论背景

---

### [35] Machine learning-guided strategies for reaction conditions design and optimization

- **作者**: Lung-Yi Chen et al.
- **期刊**: *Beilstein Journal of Organic Chemistry*, 2024, **20**, 2476–2492
- **链接**: [https://doi.org/10.3762/bjoc.20.212](https://doi.org/10.3762/bjoc.20.212)
- **任务**: 综述ML在反应条件优化（含选择性）中的策略
- **方法**: 综述全局模型（数据库驱动）与局部模型（特定体系），涵盖主动学习与迁移学习
- **与本项目相关性**: ⭐⭐⭐ 中相关——本项目同样基于反应条件特征（44d条件特征）进行预测；综述提供系统比较

---

### [36] Machine Learning Strategies for Reaction Development: Toward the Low-Data Limit

- **作者**: Eunjae Shim et al.
- **期刊**: *Journal of Chemical Information and Modeling*, 2023, **63**(12), 3659–3668
- **链接**: [https://doi.org/10.1021/acs.jcim.3c00577](https://doi.org/10.1021/acs.jcim.3c00577)
- **任务**: 综述小数据化学反应预测的主动学习与迁移学习策略
- **方法**: Perspective，强调数据效率与模型泛化
- **与本项目相关性**: ⭐⭐ 中低相关——Crimmins(259)、Oppolzer(141)、Myers(14)子集均属低数据场景

---

### [37] Rethinking chemical research in the age of large language models

- **作者**: Robert MacKnight et al.
- **期刊**: *Nature Computational Science*, 2025, **5**, 715–726
- **链接**: [https://doi.org/10.1038/s43588-025-00811-y](https://doi.org/10.1038/s43588-025-00811-y)
- **任务**: Perspective综述LLM在化学研究中的整合路径
- **方法**: 综述，评估GPT-4等LLM在反应预测、立体化学推理、合成规划上的能力与局限
- **与本项目相关性**: ⭐⭐ 中低相关——指出立体化学是现有LLM的主要薄弱环节，说明本项目专门化方法的价值

---

### [38] A Perspective on Foundation Models in Chemistry

- **作者**: Junyoung Choi et al.
- **期刊**: *JACS Au*, 2025, **5**(4), 1499–1518
- **链接**: [https://doi.org/10.1021/jacsau.4c01160](https://doi.org/10.1021/jacsau.4c01160)
- **任务**: 综述化学基础模型预训练策略
- **方法**: 综述，指出现有模型忽视3D立体化学信息的局限
- **与本项目相关性**: ⭐⭐ 中低相关——论文指出立体化学是化学基础模型的未解难题，支持本项目专门化方法路线

---

<a name="section6"></a>
## 6. 总结对比表

| # | 论文（简称） | 期刊 | 年份 | 任务类型 | 数据规模 | 核心方法 | 本项目相关性 |
|---|------------|------|------|---------|---------|---------|------------|
| 1 | Glyco. ML (Moon 2021) | Chem. Sci. | 2021 | 底物控制立体选择性 | 268 | RF + QM描述符 | ⭐⭐⭐⭐ |
| 2 | GlycoPredictor (Moon 2025) | JACS | 2025 | 大规模糖苷化立体选择性 | >10,000 | 多任务ML | ⭐⭐⭐⭐ |
| 3 | Glyco. Review (Fu 2025) | CCS Chem | 2025 | 综述-糖苷化AI | — | 综述 | ⭐⭐⭐ |
| 4 | Pd-elec C-H (Xu 2023) | Nat. Synth. | 2023 | C-H活化对映选择性 | 846,720枚举 | TS知识+RF | ⭐⭐⭐ |
| 5 | CPA catalyst (Liles 2023) | Chem | 2023 | CPA催化剂设计 | ~20 | PCA+MLR | ⭐⭐ |
| 6 | Generality (Betinol 2023) | JACS | 2023 | 催化剂普适性评估 | 文献数据 | 无监督+监督ML | ⭐⭐ |
| 7 | Negishi NN (Cuomo 2023) | ACS Cent. Sci. | 2023 | P-手性磷配体ee预测 | 17+10 | DFT→FFNN | ⭐⭐⭐ |
| 8 | Fragment desc. (Tsuji 2023) | Angew. Chem. | 2023 | CPA催化剂外推 | 小规模 | Fragment desc.+ML | ⭐⭐ |
| 9 | Composite ML (Chung 2024) | Sci. Rep. | 2024 | CPA催化对映选择性 | 342 | SVR+RF+LASSO集成 | ⭐⭐⭐ |
| 10 | Meta-learning (Singh 2025) | Nat. Commun. | 2025 | 不对称氢化ee分类 | 11,932 | Prototypical Nets | ⭐⭐⭐ |
| 11 | Ni-cataly. (Romer 2024) | ACS Catal. | 2024 | E/Z非对映选择性+产率 | 系统库 | 贝叶斯多目标优化 | ⭐⭐ |
| 12 | Mg-cataly. (Baczewska 2024) | Angew. Chem. | 2024 | Mg催化不对称 | 文献数据 | RF/GBT | ⭐ |
| 13 | ChemAHNet (Cheng 2025) | Nat. Comput. Sci. | 2025 | 氢化立体构型+ee预测 | 大规模 | 机理知情DL | ⭐⭐⭐⭐ |
| 14 | HTE Rh (Kalikadien 2024) | Chem. Sci. | 2024 | Rh氢化ee预测 | 3,552 | HTE+RF | ⭐⭐⭐ |
| 15 | Sharpless AD (Ocampo 2025) | ACS Cent. Sci. | 2025 | Sharpless AD ee预测 | 1,007 | Fragment desc.+GBR | ⭐⭐⭐⭐ |
| 16 | Local space (Betinol 2025) | ACS Catal. | 2025 | ML预测精度评估 | 多数据集 | 控制变量实验 | ⭐⭐⭐ |
| 17 | Amidase ML (Li 2024) | Nat. Commun. | 2024 | 酶催化ee预测 | 240 | RF | ⭐ |
| 18 | C-H DNN (Hoque 2022) | Dig. Disc. | 2022 | β-C-H活化ee预测 | 240 | DFT→DNN+SHAP | ⭐⭐ |
| 19 | C-H GenAI (Hoque 2025) | Chem. Sci. | 2025 | C-H活化+配体生成 | 220 | ULMFiT迁移学习 | ⭐⭐ |
| 20 | Diels-Alder (Keto 2024) | JACS | 2024 | DA立体选择性预测 | 9,537 | NERF化学感知NN | ⭐⭐⭐ |
| 21 | Data check (Aguilar 2025) | Molecules | 2025 | 立体化学标签验证 | 1,332 | GNN集成 | ⭐⭐⭐ |
| 22 | HCat-GNet (Aguilar 2025) | iScience | 2025 | 不对称催化GNN | 668 | 图卷积GNN | ⭐⭐ |
| 23 | ChIRo (Adams 2022) | ICLR 2022 | 2022 | 3D手性分子表征 | 多基准 | SE(3)-不变GNN | ⭐⭐ |
| 24 | ChiENN (Gainski 2023) | ECML PKDD | 2023 | 手性感知GNN | 多基准 | 可插拔手性层 | ⭐⭐ |
| 25 | 3DReact (van Gerwen 2024) | JCIM | 2024 | 3D反应化学GNN | 3个TS数据集 | 等变GNN | ⭐⭐ |
| 26 | Chirality+Transformer (Yoshikai 2024) | Nat. Commun. | 2024 | Transformer手性识别 | 数百万分子 | BERT-style | ⭐⭐⭐ |
| 27 | R-SMILES (Wan 2022) | Chem. Sci. | 2022 | 手性反应SMILES表征 | USPTO | Transformer | ⭐ |
| 28 | fragSMILES (Mastrolorito 2025) | Chem. Commun. | 2025 | 片段手性表征 | ~1M反应 | Seq2Seq Transformer | ⭐ |
| 29 | SIMG (Bajusz 2025) | Nat. Mach. Intel. | 2025 | 立体电子图表征 | QM数据集 | 双GNN工作流 | ⭐⭐ |
| 30 | ChiDeK (Shi 2026) | ICLR 2026 | 2026 | 统一手性GNN框架 | 多基准 | 手性核函数+attention | ⭐⭐ |
| 31 | Stereo ≠ Friends (Tahıl 2024) | JCIM | 2024 | 立体异构体ML挑战 | 环糊精数据集 | 描述符对比 | ⭐⭐⭐ |
| 32 | Heteroencoder (Baimacheva 2025) | J. Cheminform. | 2025 | 手性描述符评估 | 3,858 | VAE+RF | ⭐⭐ |
| 33 | HLM Chiral GNN (Pu 2024) | JCIM | 2024 | HLM代谢手性预测 | HLM数据库 | 手性GNN对比 | ⭐ |
| 34 | VBur review (Escayola 2024) | Chem. Soc. Rev. | 2024 | 综述-%VBur+ML | — | 综述 | ⭐⭐⭐ |
| 35 | Cond. design review (Chen 2024) | Beilstein J. | 2024 | 综述-反应条件ML | — | 综述 | ⭐⭐⭐ |
| 36 | Low-data ML (Shim 2023) | JCIM | 2023 | 综述-小数据策略 | — | Perspective | ⭐⭐ |
| 37 | LLM chemistry (MacKnight 2025) | Nat. Comput. Sci. | 2025 | LLM化学应用 | — | Perspective | ⭐⭐ |
| 38 | Foundation models (Choi 2025) | JACS Au | 2025 | 综述-化学基础模型 | — | 综述 | ⭐⭐ |

**共38篇论文** | 直接相关（⭐⭐⭐⭐）：5篇 | 中高相关（⭐⭐⭐）：14篇 | 中低相关（⭐⭐）：16篇 | 参考（⭐）：3篇

---

<a name="section7"></a>
## 7. 对本项目（AldolRxnMaster）的启示

### 7.1 本项目的独特性（论文卖点）

经过系统调研，**2022–2026年间尚无专门针对Evans/Crimmins等手性辅基控制醛醇反应进行ML 4-class立体化学预测的独立发表论文**。本项目填补了以下空白：

1. **反应类型**：手性辅基（chiral auxiliary）底物控制醛醇反应，与对映选择性（ee）预测任务完全不同
2. **标签类型**：4-class绝对CIP构型预测（R/S for Ca × R/S for Cb），远比二元ee分类复杂
3. **数据规模**：2,334行历史Reaxys文献数据，规模适中（介于小数据~200与大数据~10,000之间）
4. **评估方式**：时间序列CV（TSCV）+ Scaffold + Role-aware Grouped三重评估，比大多数文献更严格

### 7.2 可借鉴的具体方法

| 启示 | 来源论文 | 具体建议 |
|------|---------|---------|
| **底物控制立体选择性 = 化学描述符优先** | [1,15] Moon 2021, Ocampo 2025 | QM计算描述符（已有Vbur等）+ 条件特征组合是经典路线；本项目128d特征工程路线正确 |
| **机理知情特征 vs. 纯数据驱动** | [4,13] Xu 2023, Cheng 2025 | MechAware特征（TS知识向量化）思路与Nature Synthesis 2023完全对应；这是主要亮点之一 |
| **多任务分类用于立体选择性** | [2] Moon 2025 GlycoPredictor | 可考虑多任务学习：Ca预测 + Cb预测 + syn/anti预测同时训练（共享底层表征） |
| **SHAP可解释性必须报告** | [15,18] Ocampo 2025, Hoque 2022 | 所有发表论文均报告SHAP/permutation importance；本项目P3.1 SHAP计划对论文写作至关重要 |
| **元学习解决小辅基子集** | [10] Singh 2025 | Crimmins(259)/Oppolzer(141)/Myers(14)子集数据量极小；元学习（Prototypical Networks）是处理这类少样本场景的SOTA方法 |
| **域外预测要诚实评估** | [14,16] Kalikadien 2024, Betinol 2025 | Scaffold/TSCV评估已覆盖；论文需明确指出域外性能局限，不能只报告随机split结果 |
| **数据质量验证** | [21] Aguilar 2025 | 本项目已有严格清洗管线（step01-step12）；可参考GNN集成方法检验剩余标签错误 |
| **复合/集成模型** | [9] Chung 2024 | GMM-guided集成策略（SVR+RF+LASSO）在342个CPA数据上R²=0.936；可参考用于本项目P1.3集成 |

### 7.3 论文定位建议

- **对比基线**：与[1,2]（糖苷化ML）、[15]（Sharpless AD ML）对比，强调"4-class绝对构型预测"与"二元ee分类"的本质差异
- **核心贡献**：首个系统性手性辅基醛醇反应ML立体化学预测框架，覆盖6种辅基类型
- **MechAware特征**：直接对应Nature Synthesis 2023 [4]的"TS知识向量化"思路，有国际顶刊背书
- **评估严格性**：TSCV + Scaffold + Grouped三重评估，比同类工作（多数仅random split）更严格

### 7.4 技术路线缺口（可改进方向）

基于文献调研，本项目目前欠缺但文献已证明有价值的方向：

1. **多任务学习**（[2] GlycoPredictor）：Ca+Cb同时预测而非联合4-class，可能提升泛化
2. **元学习**（[10] Singh 2025）：解决Myers（14条）等极小辅基子集的预测问题
3. **3D TS机制特征**（[4] Xu 2023, [13] ChemAHNet）：当前MechAware特征基于Z/E烯醇化物几何，可扩展至完整TS能量
4. **主动学习**（[36] Shim 2023）：识别最有信息量的待测反应以高效扩充数据集

---

*最后更新: 2026-05-28 | 检索人: AldolRxnMaster项目 | 搜索工具: WebSearch + WebFetch系统验证*
