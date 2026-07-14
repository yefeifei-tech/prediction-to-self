# Three Levels of Learning — Working Notes

**状态**：pre-outline notes（still gathering insights，not yet paper draft）

**核心 claim**：机器学习不是"知识创造"，是"内部动力学与世界结构的对齐重塑"。Reservoir Computing 和 Deep Learning 不是对立范式，是同一个三层框架的不同层级选择。

---

## 三层框架

| Level | 名称 | 做什么 | 例子 | 训练什么 |
|:---:|------|--------|------|---------|
| L1 | **Dynamics** | 产生丰富的状态轨迹 | 天气系统、随机 GRU、任何 stateful dynamical system | 无（"给定"的） |
| L2 | **Readout** | 从状态中恢复任务相关信息 | Prediction head、linear probe、W_action | 只训 readout |
| L3 | **Dynamics Shaping** | 重塑状态空间使表征更紧凑、稳定、可组合、易读 | 现代深度学习（GPT、Transformer 全网络训练）| 训全部权重 |

**Level 是累加的**：L2 包含 L1，L3 包含 L1 + L2。

## 关键 reframe

**旧的定位**（传统 DL 叙事）：训练 = 学习 = "系统变聪明"

**新的定位**（三层框架）：**训练 = 让系统的内部动力学与世界的动力学结构对齐得更好**

**具体后果**：
- **"聪明"是 readout 层的能力**（发现有用子空间）
- **"匹配"是 shaping 层的能力**（重塑内部结构与世界结构对齐）
- 两者不同层次，不应混淆

## 为什么这个 reframe 深刻

**Reservoir Computing 有效的深层理由**：**因为它选择了和世界同类的数学对象**（连续动力学系统），**所以自然产生"共振"**——不需要显式学习 dynamics，随机 dynamics 本身就在物理层面对齐世界。

**世界和 GRU 属于同一种数学对象**：
- 世界 = 动力学（天气、股票、语言 token 序列——全部是连续变化）
- GRU / RNN / Mamba = 动力学

**Representation ≠ Encoding，Representation = Ontological Alignment**（本体论对齐）：
- 传统 ML 视角：模型**编码**世界
- 三层视角：模型**是**世界的一种同构变体

**这个 shift 有一个操作性后果**：**"表征"的可能性来自"世界和模型都是同类数学对象"这个前提**。**符号 AI (GOFAI) 失败的根本原因**——它选择了离散符号，与连续世界属于不同数学对象，无法自然对齐。

## Paper 1 在这个框架里的位置

**Paper 1 是 L2 的最小 empirical demonstration**：
- **GRU 冻结**（不做 L3 shaping）→ 只使用随机 dynamics
- **pred / W_action 训练**（做 L2 readout）→ 从 dynamics 中发现有用子空间
- **AG > 0、encoding gap、encoding breakthrough** 全部在 L2 层解释

**Paper 1 停在 L2 是刻意选择**（为了 attribution 干净），**不代表 L2 是唯一或最优方式**。

## 三层框架的可测预测

**Prediction 1（stateless fails）**：**没有 L1（stateful dynamics）的架构**无法通过 reservoir + readout 机制产生 self-representation
- **失败**：vanilla Transformer, MLP（stateless）
- **成功可能**：RNN, Mamba, KV-cached Transformer, RWKV, xLSTM（stateful）
- **实验设计**：在 Paper 1 的 causal-loop setup 上换 stateless 架构，看 encoding gap 相关指标

**Prediction 2（L2 vs L3 的 self-representation 结构差异）**：**L3 shaping 后的 self-representation 应该比 L2 reservoir 更结构化**
- L2（Paper 1 reservoir）: self 表征是"h 里的一个 readable 方向"——**扁平的**
- L3（训 GRU 版本）: self 表征可能是"层级化的、可组合的、跨任务迁移的"
- **实验设计**：用同一套 probe 测量 L2 和 L3 版本，看**表征的结构性质**（不只是 recall 数字）

**Prediction 3（shaping 的必要性 = 任务复杂度）**：**任务越复杂，越需要 L3 shaping**
- 简单动力学预测（正弦、Lorenz）→ L2 reservoir 足够
- 需要层级概念（语言、视觉）→ 必须 L3

## 需要精细化的一处

**"学习 = 压缩"这个具体表述太窄**。

**反例**：
- Kernel methods 里训练做"扩展"（低维 → 高维）
- VAE / Diffusion 里训练做"分布重塑"
- Metric learning 里训练做"分离"
- Word embedding 里训练做"聚合"

**更 general 的说法**：**训练 = 结构性重塑（Structural Reshaping）**，具体机制可以是**压缩、扩展、分离、聚合、混合**，视任务而定。

**"压缩"是最常见的一种**（Information Bottleneck 视角），但不是唯一。

## 与 CET 的对应关系

**这个三层框架不是独立于 CET 的**——**它是 CET 观察者视角的 mechanism-level 实现**：

| CET 层面 | 三层框架对应 |
|---------|-------------|
| 约束（客观 C）| L1 Dynamics 提供的原始表征空间 |
| 信息（主观 I）| L2 Readout 从中读出的部分 |
| 学习（吸收约束）| L3 Shaping：让约束的组织形式更易读 |

**"信息不是创造约束，而是恢复约束"（CET）** ↔ **"表征不是创造出来的，是从动力学中读出来的"（三层框架）**

**这个平行不是巧合**——**两者是同一哲学立场在不同层面的表达**。

## Related Work（初步）

**Reservoir Computing 的经典文献**：
- Jaeger, H. (2001). "The 'Echo State' Approach to Analysing and Training Recurrent Neural Networks."
- Maass, W., Natschläger, T., Markram, H. (2002). "Real-time computing without stable states: A new framework for neural computation based on perturbations."

**Deep Learning 的表征学习视角**：
- Bengio, Y., Courville, A., Vincent, P. (2013). "Representation Learning: A Review and New Perspectives."
- 大量后续工作

**Ontological Alignment / Enactivism**（哲学基础）：
- Varela, F., Thompson, E., Rosch, E. (1991). "The Embodied Mind: Cognitive Science and Human Experience."
- Clark, A. (2016). "Surfing Uncertainty: Prediction, Action, and the Embodied Mind."

**信息瓶颈 / 压缩视角**：
- Tishby, N., Zaslavsky, N. (2015). "Deep learning and the information bottleneck principle."

**尚未查文献但需要查的**：
- Neural ODEs（Chen et al., 2018）——同样"世界是 dynamics，模型也是 dynamics"的立场
- Predictive Coding literature（Friston, Rao & Ballard）
- Embodied AI / Sensorimotor coupling

## Open Questions（待思考）

1. **三层是"严格的架构分层"还是"分析视角"？** —— 现代 DL 里 attention 同时做 shaping 和 readout，三层不一定是干净的模块。
2. **"本体论对齐"的可测化** —— 如何 quantify"两个动力系统的对齐程度"？可能需要动力系统理论（吸引子距离、耦合系数等）。
3. **L3 shaping 到底改变了 dynamics 的什么** —— 谱结构？吸引子拓扑？流形形状？这需要更精细的 dynamical systems analysis。
4. **Level ↔ Emergence 的关系** —— 更高 Level 是否**必然**产生更强的 emergence（self-representation、abstraction、compositionality）？还是只是 correlated？

## Discussion Timeline

**2026-07**（Paper 1 讨论中 emerge）：
- Reservoir Computing 事实揭示（GRU 冻结，只训 readout）
- 从"学习 = 创造知识"转向"学习 = 发现/选择"
- 三层框架被 articulate 出来
- 与 CET 的哲学立场对齐被识别

**下一步**：待定
- 或深化理论（更多哲学论证）
- 或跑 empirical 实验（Prediction 1/2/3）
- 或直接开始 outline / draft
