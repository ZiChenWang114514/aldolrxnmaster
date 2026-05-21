# AldolRxnMaster — Roadmap

## 项目状态 (2026-05-16)

- **数据**: V3 重建 (4751 → 1655 Evans, 16 步严格清洗)
- **冠军**: MechAware-Full (151d), TSCV = **0.733 ± 0.074**
- **基准**: 15 active models 在 V3 数据上完成公平评估
- **泄漏修复**: DRFP 标签泄漏 + 旧 group_id 泄漏 已确认并排除

---

## 已完成 ✓

- [x] **Phase A**: 数据清洗 (1822 → 1801 → V3 1655)
- [x] **Phase B**: 图表示构建 (4 种 PyG 图)
- [x] **Phase C**: GNN 实验 (12 组合, 全部失败, 最佳 0.497)
- [x] **Phase D**: 特征融合 (DRFP/RXNFP, 全部负面)
- [x] **V3 数据重建**: 16 步管线, SA 一致性, 角色保留去重
- [x] **MechAware 模型**: Z/E 烯醇盐分离构象 + base 加权 → TSCV 0.733
- [x] **公平评估**: 同数据同划分, 泄漏检测 (scaffold -7%, grouped -10%)
- [x] **项目整理**: 统一命名, 归档旧代码, MODEL_REGISTRY, 目录重组
- [x] **DRFP 泄漏调查**: 确认产物 @/@@ 编码答案

---

## 下一步

### Publication Preparation
- [ ] 撰写论文 (JACS / Nature Comms)
- [ ] 主要 story: MechAware Z/E 机制感知特征 > 所有端到端方法
- [ ] 补充: 泄漏检测方法论 (对 community 有警示价值)
- [ ] Figure: TSCV 4-fold 对比图 + SHAP feature importance

### 模型改进方向
- [ ] Ensemble: MechAware-Full + V2-XGB stacking
- [ ] 修复 DRFP: 去除产物 @/@@ 后重新评估
- [ ] 更多数据: 非 Evans 迁移学习 (430 行 non-Evans 备用)
- [ ] 类别不平衡: anti 类 (C1=10%, C2=7%) focal loss / SMOTE

### 代码质量
- [ ] 单元测试覆盖核心函数
- [ ] CI/CD 配置
- [ ] API 文档 (chiralaldol/ 模块)
