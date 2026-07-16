# Trace Ablation Experiment — pending

**状态**：Design draft，代码未实现（**Pending** —— 等有空再做）

**来源**：2026-07 讨论 exp3/exp4 时 emerge——**Paper 1 exp4 claim"1D trace 是最小充分构造"，但没做严格 ablation 验证"必须是 trace"**。

---

## 动机

**Paper 1 现状**：
- **exp3**：4d GRU 输入（obs only）→ trailing recall ≈ **12%**
- **exp4**：5d GRU 输入（obs + τ）→ trailing recall ≈ **60%+**
- 结论：**加一维 trace 就能跨过 encoding gap**

**未被严格证明的边界**：
- **是"trace 特异性"起作用**，还是**"任何一维额外输入 + 合适 pretext task"都行**？
- 如果是后者，"1D trace = 最小充分构造"就弱化为"1D 额外输入 = 提升 recall 的通用手段"——**大幅削弱 Paper 1 §3.4 的论证力度**

**Paper 1 §7 limitation 提到过这类问题**，但没显式测过 trace-specific vs generic-1D-input 的对照。

## 为什么这个 ablation 关键

**如果结果显示"只有 trace 有效"** →
- **强化** "trace 特异性"这个 claim
- **强化** "self-history 需要专门的、action-derived 的通道"这个理论 claim
- **强化** proprioception 类比的严格性

**如果结果显示"任何 1D 输入都能类似提升"** →
- **弱化** encoding gap 的架构性 claim
- **暗示** encoding gap 本质是"信息容量/维度"问题，不是"信号特异性"问题
- **修正** Paper 1 §3.4 的表述——从"trace 特异"改为"1D 额外通道"

**无论哪种结果都有科学价值**——都精细化了 Paper 1 的边界。

---

## 三个具体 Ablation

### Ablation 1：随机噪声通道（最重要）

**Setup**：与 exp4 完全相同，但 τ 替换为**独立同分布的 uniform 或 gaussian 噪声**：

```python
# 原 exp4：
self.trace = TRACE_BETA * self.trace + (1 - TRACE_BETA) * abs(action_val)

# Ablation 1：
self.trace = np.random.uniform(-1, 1)   # 每步 iid 随机，与 action 无关
```

**问题**：**纯噪声通道能否提升 trailing recall？**

**预测**：
- 如果 recall 仍然 ≈ 60%+ → **1D 额外通道本身有效，不是 trace 特异**
- 如果 recall ≈ 12%（同 exp3）→ **只有 trace 有效**

### Ablation 2：AR(1) 相关噪声通道

**Setup**：τ 替换为 AR(1) 过程，**统计上有时间结构但和 action 无关**：

```python
# Ablation 2：
self.trace = 0.95 * self.trace + 0.05 * np.random.randn()   # AR(1) noise
```

**问题**：**"有时间结构但内容无关"的信号能否提升 recall？**

**预测**：
- 如果 recall ≈ 60% → **时间结构本身就够，内容不重要**
- 如果 recall ≈ 12% → **必须内容和 action 相关**
- 如果 recall 介于两者 → **两者都贡献**

### Ablation 3：Trace 的 EMA 时间常数扫描

**Setup**：保持 τ = EMA(|action|)，但扫描 β：

```python
# β 扫描：
for beta in [0.5, 0.8, 0.9, 0.95, 0.99]:
    self.trace = beta * self.trace + (1 - beta) * abs(action_val)
```

**问题**：**recall 对 EMA 时间尺度有多敏感？**

**预测**：
- 太快衰减（β=0.5，半衰期 ≈1 步）→ trailing 期间 τ 早已衰减到 0，recall 应下降
- 太慢衰减（β=0.99，半衰期 ≈69 步）→ trailing 和 quiet 都保持痕迹，recall 也应下降
- **有一个最优 β 附近**（应该在 0.95 附近，因为 trailing 窗口 50 步）

**如果确实是钟形曲线** → 强化 "trace 的时间尺度和 trailing 窗口对齐是关键" 这个 mechanism claim

---

## 实现要点

**代码基础**：直接 fork [experiments/exp4_proprioception.py](../../experiments/exp4_proprioception.py)，改**只需要改 `update_trace` 一处**（`ProprioceptiveModel` 类的方法）。

**架构**：完全同 exp4，除 τ 的生成规则外。

**训练**：完全同 exp4（100K Phase 1 + 80K Phase 2，同 LR / GAMMA / aux weight）。

**Seed**：与 exp3/exp4 保持一致（seed=42），保证唯一变量就是 τ 的定义。

**输出**：跑三种 ablation，每种输出 trailing recall + comparison table：

```
Baseline (exp3, no trace):        recall ≈ 12%  (references only)
Original trace (exp4):            recall ≈ 60%+ (references only)
Ablation 1 (random noise):        recall = ?
Ablation 2 (AR(1) noise):         recall = ?
Ablation 3 (β sweep):
  β=0.5:  recall = ?
  β=0.8:  recall = ?
  β=0.9:  recall = ?
  β=0.95: recall = ?  (same as exp4)
  β=0.99: recall = ?
```

**大概要跑的时间**：每组约 20 分钟（借鉴 exp4 quick mode 时间），全套 3+5 = 8 组 → 约 3 小时。

---

## 与其他工作的关系

**与 Paper 1**：**这是 Paper 1 exp4 缺失的严格 ablation**——不是新实验，是 Paper 1 §7 limitation 隐含承认的洞的填补。

**与 papers/three_levels/**：Ablation 结果可能会影响三层框架里"L2 sufficient 条件"的表述——如果只有特定信号有效，L2 的门槛比 "任何 1D 额外输入"更严格。

**与 experiments_ext/metacog_level4/**：**这两个都是 exp7+ 方向的 pending experiments**。metacog_level4 探测 Level 3→4（递归自我模型），trace_ablation 精细化 Level 2→3（encoding gap 突破的严格边界）。

## 状态

**Pending**：
- 设计 draft ✓（本文档）
- 代码实现 ✗
- 跑实验 ✗
- 结果分析 ✗

**触发条件**：
- Evan 有 3 小时空档
- 或有具体的写作/审稿场景需要这个 ablation 结果做支撑
- 或 exp7 metacog_level4 恢复推进时，一起把这些"cleanup ablations"做了

**优先级**：**中等**——不阻塞 Paper 1 已发布内容，但会显著加强未来的严谨性讨论（尤其对 ML 受众）。

## 参考

- **来源讨论**：2026-07 exp3/exp4 深读会话
- **相关手册章节**：`docs/deep_reads/exp3_exp4_encoding_gap.md` §3.4（exp4 严格边界的第 2 条）
- **Paper 1 相关**：Section 3.4（Proprioceptive Breakthrough），Section 7（Limitations）
