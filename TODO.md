# AldolRxnMaster — Roadmap

## 项目状态 (2026-05-27)

- **数据**: V4d 管线从 134K Reaxys 原始数据重建，**2334 行**（6 种辅基类型）
- **冠军**: **ma_bw_xgb** (156d), TSCV = **0.625 ± 0.040**, Grouped = **0.773 ± 0.024**
- **3D syn/anti**: step08b 3D 二面角法计算，98.7% 成功率，仅作分析标签
- **基准**: 11 active models 在 V4c 数据(2288行)完成评估；V4d(2334行)重新基准进行中
- **管线可复现**: 13 步清洗(含 step08b) + 行级审计，完全从原始 Reaxys 数据出发

---

## 已完成

- [x] **V4 数据重建**: 134K Reaxys → 13 步清洗 → 2334 行 (6 种辅基)
- [x] **辅基检测**: Evans + Crimmins thione/oxathione + Oppolzer + Myers + generic
- [x] **手性催化排除**: proline/BINAP/cinchona 等手性催化剂自动排除
- [x] **CIP 标签提取**: 产物 SMILES → atom mapping → Cα/Cβ CIP 编码
- [x] **3D syn/anti 修复**: step08b 3D 二面角法替代不可靠的 CIP 启发式(~52%)
- [x] **保护 OH 模板扩展**: step06 新增 6 条 SMARTS (silyl/benzyl/acetal)，+46 行数据
- [x] **V4 特征工程**: 构象(ETKDGv3) + steric(34d) + conditions(44d) + aux(6d) + chirality(7d) + R-group(8d) + ChiralEnv(21d) + AldPriority(8d) = 128d
- [x] **MechAware V4c**: Z/E 分离 + BW 加权，champion ma_bw_xgb TSCV=0.625
- [x] **V4c 基准**: 11 models × 10 splits = 110 predictions (2288 行)
- [x] **DRFP 泄漏确认**: 产物 @/@@ 编码答案 → 排除
- [x] **Mukaiyama 过度排除修复**: step03 对有辅基的行不用命名反应排除

---

## 进行中

- [ ] **V4d 管线刷新**: features/splits/mechaware/models 全部基于 2334 行重新生成
- [ ] **全文档更新**: CLAUDE.md + TODO.md + LESSONS.md + RESULTS.md 更新到 V4d 现状

---

## 下一步

### 模型改进
- [ ] Ensemble: ma_bw_xgb + v4b_full_et + ma_full_xgb stacking
- [ ] 辅基感知建模: 不同辅基类型可能需要不同的 steric 描述符权重
- [ ] Optuna 超参优化: XGBoost/LightGBM/ET 系统调参
- [ ] 多构象特征: 用构象集成替代单构象 steric 描述符

### Publication Preparation
- [ ] 论文撰写
- [ ] Story: 可复现的底物控制 aldol 数据集 + 3D syn/anti 方法 + steric 特征 > 端到端方法
- [ ] 泄漏检测方法论 (DRFP 案例)
- [ ] Figure: benchmark 对比 + SHAP + 辅基类型分析 + 二面角分布
