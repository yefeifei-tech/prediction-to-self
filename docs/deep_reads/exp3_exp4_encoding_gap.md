# exp3 + exp4：Encoding Gap 深读

**用途**：单独深读 Paper 1 中 exp3 (Encoding Gap) 和 exp4 (Proprioceptive Breakthrough) 两个实验。**必须合起来读**，因为只有对照才能理解 encoding gap 概念。

**前置阅读建议**：先看 [paper1_解读手册.md](../paper1_解读手册.md) §7 分实验详解里的 exp2 章节，理解 implicit causal utilization 的证明结构后再读这份文档。

---

## 目录

1. [前置：exp2 引出的问题](#1-前置exp2-引出的问题)
2. [exp3 深读：Encoding Gap 的存在证据](#2-exp3-深读encoding-gap-的存在证据)
3. [exp4 深读：Encoding Gap 的最小充分构造](#3-exp4-深读encoding-gap-的最小充分构造)
4. [合读：Encoding Gap 作为核心概念](#4-合读encoding-gap-作为核心概念)
5. [未解决的问题](#5-未解决的问题open-questions)
6. [附录：CET 视角的重解读](#6-附录cet-视角的重解读)

---

## 1. 前置：exp2 引出的问题

### exp2 建立了什么

exp2 用行为学证据证明：**系统能通过 causal loop 做预测**（Implicit Causal Utilization）——三层证据链：
- Spike test（通道特异性）
- Recovery test（结构化因果 vs 统计关联）
- Self-maintenance（选择效应）

### exp2 留下什么问题

**exp2 全部用行为层证据**（预测质量、恢复速度）——**不直接看 h 内部**。这留下一个关键 open question：

> **系统能利用因果通路做预测（exp2 证），是不是意味着 h 里有可读的"我在动"表征？**

这是**两种截然不同的可能**：

| 可能 | 内部机制 | 含义 |
|-----|---------|-----|
| **A（强 claim）** | h 里有一个可读的维度编码 self 状态 | 系统"知道"自己在动 |
| **B（弱 claim）** | 信息弥散在 h 的多个维度里，pred 通过复杂线性组合利用 | 系统只是"会做"，不"知道" |

**exp2 的证据兼容两者**——recovery/spike 都不能区分 A 和 B。

### 打开黑箱：需要什么

**要区分 A 和 B**，必须**直接看 h**：
- 训一个 **probe**（探针）从 h 读"是否在动"
- 如果 probe 能读出（recall 高）→ **A 成立**
- 如果 probe 读不出（recall ≈ chance）→ **A 不成立，只有 B**

**exp3 就是设计来测这个的**。

---

## 2. exp3 深读：Encoding Gap 的存在证据

### 2.1 Setup

**文件**：[experiments/exp3_encoding_gap.py](../../experiments/exp3_encoding_gap.py)

**架构**（关键：**没有 trace**）：
- GRU 输入 = **obs (4d) only**（没有 action 也没有 trace）
- **Aux head**：3 分类（active / trailing / quiet），梯度**打进** h
- **BinaryProbe**：从 **detached h** 读 trailing 二分类

**训练模式**（详见手册 §6 训练模式对照）：
- **Phase 1**（100K 步）：与 exp2 同——**reservoir**，只训 pred
- **Phase 2**（80K 步）：加入 aux head，**梯度通过 h_live 流回 GRU**——pred + GRU + aux 都被训练

Phase 2 用 `step_live`（h_live 不 detach），Phase 1 用 `step_frozen`（h_multi.detach）。**这个训练模式差别很关键**：Phase 2 里 GRU 是被训的，不再是 reservoir。

### 2.2 三个关键设计（每个都值得深挖）

#### 设计 1：**Action 必须外部随机**

**代码位置**：[exp3_encoding_gap.py:170](../../experiments/exp3_encoding_gap.py#L170)

```python
action_val = np.random.uniform(-ACTION_RANGE, ACTION_RANGE) if (ba or is_p1) else 0.0
```

**关键点**：exp3 里 action **不是 W_action(h)**（不像 exp2 causal 组），是**纯随机 `uniform(-2, 2)`**。

**为什么这么设计**？

**如果 action = f(h)**（如 exp2 causal）：
- action 是 h 的确定性函数
- h 里天然编码 action（因为 h → action 是一个映射）
- **probe 读 h 就等于间接读 action** → **平凡地 100% 成功**
- 但这**不代表本体感觉存在**——只代表"运动指令还在 h 里"

**Paper 1 的类比**：
- **判断一个患者有没有本体感觉**，不能问"你知道自己要抬腿吗"（他知道，因为他自己发指令）
- 要问"**你能感觉到自己的腿被抬起来了吗**"（这才测的是感觉，不是意图）

**exp3 的随机 action 就是"实验者替系统抬腿"**——切断"运动指令 shortcut"，让 probe 只能靠**真实的本体感觉信号**（obs 反馈或 trace）才能识别 self 状态。

**这个设计是 encoding gap 能被干净测出的前提**——如果用 exp2 那种 W_action(h)，probe 会平凡成功，encoding gap 概念根本无法测量。

#### 设计 2：**Aux head 和 Probe 的分工**

**代码位置**：Aux head [exp3:107](../../experiments/exp3_encoding_gap.py#L107)；Probe [exp3:115](../../experiments/exp3_encoding_gap.py#L115)

两者都是从 h 读信息，但**目的完全不同**：

| | Aux head | Probe |
|---|--------|-------|
| **输入** | h_live（**未 detach**）| h_multi.detach（**detach**）|
| **输出** | 3 分类（active/trailing/quiet）| 二分类（is_trailing）|
| **梯度是否回流 h** | **是**（塑造 h）| **否**（只测量 h）|
| **角色** | **训练时施压**——逼 h 编码 | **测量时读取**——看 h 里有什么 |

**类比**：
- **Aux head 像每天督促你吃药**——它主动改变你的身体状态
- **Probe 像给病人做 X 光**——它测你身体里有什么，自己不改变什么

**关键设计意图**：**Aux 使劲推，Probe 被动读。如果 Probe 读不出，说明"h 里根本没有可读表征"（不是训练不够）**。

**这是 encoding gap 的严格定义**：**训练施压都推不出可读表征** = **架构性缺陷**。

#### 设计 3：**BurstGate + Trailing 时段**

**代码位置**：[exp3:54](../../experiments/exp3_encoding_gap.py#L54) BurstGate 类

**机制**：
```
时间 →
├─── active ───┤─── quiet ───┤─── active ───┤...
    action 随机开关
```

**三个类别定义**：
- **active**：正在 burst，action ≠ 0
- **trailing**：burst 刚结束后 **50 步**，action = 0 但**刚刚动过**
- **quiet**：长期静默，action = 0 且**很久没动过**

**为什么 trailing 是最关键测试**？

看 obs 的表现：
- **active**：obs[0] 被 action 扰动 → **obs 看起来不一样** → 系统一看 obs 就知道
- **quiet**：obs[0] 平静，长期无扰动 → **obs 看起来平静** → 系统一看也知道
- **trailing**：action 已经变 0，obs[0] 已经不再被扰动 → **obs 看起来和 quiet 完全一样**

**关键**：**要在 trailing 时段识别出"我刚动过"，必须靠内感觉（h 里的记忆）**——**obs 反馈已经失效**（obs 平静了）。

**这就是 encoding gap 的核心测试**：
- Active vs quiet：容易（obs 差异明显）
- **Trailing vs quiet：obs 看起来一样，只能靠 h 内部记忆区分**
- **如果 h 里没有可读的 self-history 维度，trailing 就识别不出来**

### 2.3 结果与解读

**关键数字**：**trailing recall ≈ 12%**（接近 chance）

**Recall 的计算**（见手册 §7 exp3 章节的 "trailing recall 是怎么计算出来的"）：
```
recall = TP / (TP + FN) = probe 说 trailing 且实际 trailing 的次数 / 实际 trailing 的总次数
```

**只统计 trailing 时段**，因为：
- 类别极度不平衡（trailing 只占 ~6%）
- Accuracy 陷阱：永远猜"not trailing"就能 94% accuracy
- Recall 才是能力测试

**12% 意味着**：probe **绝大多数 trailing 时段都错过了**（88% 漏诊）——**h 里没有可被线性 probe 读出的"我在动过"信号**。

### 2.4 严格边界

**exp3 单独证明了什么**：**在这个具体架构 + 训练量下，h 里没有可被线性 probe 读出的 trailing 信号**。

**exp3 没有证明**：

1. **不能证明"永远无法有可读表征"**——只证明**这个具体 setup**下没有
2. **不能排除"更复杂的（非线性）probe 或许能读到"**——**Paper 1 只测线性 probe**
3. **不能排除"更长训练能读到"**——只测了 80K 步 Phase 2
4. **不能证明 h 里完全没有 self 信息**——只证明信息不是"线性可分的独立维度"（可能弥散地存在）

### 2.5 常见误读

**误读 1**："exp3 是负面结果，价值有限"
- **错**。negative claim（"读不出"）在科学上**比 positive claim 更强**——只需要证否一次就足够。
- **exp3 的价值不在"发现"encoding gap（这从 ML 视角看是分布式表征的常识），而在"为 exp4 建立对照 baseline"**。**必须先证明"没有可读表征"，exp4 的"加 trace 就有可读表征"才有对照力度**。

**误读 2**："probe 读不出因为 probe 太弱（线性太弱）"
- **错**。pred 也是线性 Linear(192, 4)，能预测得好——**同类型 readout 里 pred 成功、probe 失败**，说明**不是 readout 层能力问题，是 h 里没这信息可读**。

**误读 3**："probe 读不出因为训练不够"
- **错**。aux head 使劲推 h 编码 trailing（梯度 80K 步一直在打），probe 还是 12%。**训练施压都推不出，说明是架构性缺陷**。

**误读 4**："action 用随机的没意义，应该用 W_action(h) 才是自主 action"
- **错**。**恰恰因为 action 是外部随机**，probe 才能测**纯本体感觉**（内感觉）——不是"运动意图残留"。如果用 W_action(h)，probe 会平凡成功（因为 h 里天然含 action 指令）。

---

## 3. exp4 深读：Encoding Gap 的最小充分构造

### 3.1 Setup（与 exp3 完全对照）

**文件**：[experiments/exp4_proprioception.py](../../experiments/exp4_proprioception.py)

**与 exp3 的差异**：**只有 GRU 输入维度**（4d → 5d）
- exp3: GRU 输入 = obs (4d)
- exp4: GRU 输入 = obs + **τ** (5d)

**其他一切完全相同**：
- Action 依然是随机（`np.random.uniform(-2, 2)`）——**没变**
- BurstGate、Aux head、Probe、训练步数、learning rate——**全部完全相同**
- Phase 1 / Phase 2 训练模式——**完全相同**

**这个"最小对照"设计是 exp4 结论力度的关键**——**只改一处架构**，其他所有变量都 hold constant。**任何差异都只能归因于这一处改动**。

### 3.2 τ (trace) 的关键性质

**定义**（[exp4:87](../../experiments/exp4_proprioception.py#L87)）：
$$τ_t = 0.95 \cdot τ_{t-1} + 0.05 \cdot |a_t|$$

即：**action 幅度的指数移动平均**（EMA），衰减系数 β = 0.95。

**三个关键性质**：

#### 性质 1：**只有一维**
- 单个标量信号，占 GRU 输入的 1/5
- **不是复杂多维信号，就一个数**
- 关键结论"1D 就够"的最小性来自这里

#### 性质 2：**内源**（endogenous）
- **完全从 action 自身派生**——`|a_t|` 是 action 的绝对值
- **不需要外部 label**
- 不携带 is_trailing 标签，也不携带 burst 状态

**这个"内源"性质是 encoding gap 论证的关键**：
- **如果 trace 需要外部 label**（比如直接告诉系统"你在 trailing"），那 exp4 结论就 trivial 了（给答案当然能读）
- **trace 只是 action 的 EMA**——**系统只能从 EMA 里推断 self 状态**，需要**学习**才能利用

**类比**：**trace 就是本体感觉的最小工程实现**——生物本体感觉本质就是"肌肉活动的衰减信号"，肌肉纺锤感受器持续报告"肌肉张力变化"。exp4 的 trace 用 1 维 EMA 抽象了这个信号。

#### 性质 3：**衰减保留时间信息**
- Active 时：τ 累积上升
- Burst 一停：τ **慢慢衰减**（不是立刻归零）
- Trailing 时段：**τ 还很高**，quiet 时段 τ 已经衰减到 0
- **这就是 trailing 和 quiet 可区分的物理基础**

**具体数字**：β = 0.95，衰减半衰期约 **14 步**（$0.95^{14} \approx 0.49$）——trailing 窗口 50 步内 τ 从 ~1.5 衰减到 ~0.5，**仍然明显高于 quiet 的 ~0**。

### 3.3 结果与解读

**关键数字**：**trailing recall ≈ 60%+**（从 exp3 的 12% 跳升到 60%+）

**5 倍提升**——**从 chance 附近跳到显著超越**。

**为什么这么大跳升**？

对比 h 在 exp3 和 exp4 里的信息内容：
- **exp3**（obs 4d 输入）：h 里的 self 信息**来自 obs 反馈**——active 期间 obs[0] 被扰动，进入 h 有痕迹；但 trailing 期间 obs 平静，痕迹被后续 quiet obs 覆盖，**快速衰减**
- **exp4**（obs + τ 5d 输入）：h 里的 self 信息**多了一个专属通道** τ ——τ 在 trailing 时段**仍然明显**，直接进入 h 作为可读维度

**exp4 加的不是"信息量"，是"信息组织的可读性"**——action 的信息在 exp3 里也**存在**（obs 反馈过），只是**不是可被线性 probe 读的形式**（分布式弥散）。**τ 给了 h 一个"专用通道"，让 self 信号在 h 里形成可读维度**。

### 3.4 严格边界

**exp4 证明了什么**：**加一维内源信号 τ，能让 h 里出现可被线性 probe 读出的 self 表征**。

**exp4 没有证明**：

1. **"1D 是理论最小"** — 这是 **empirical 观察**，不是严格证明。理论上可能更小（比如 sparse 编码）。
2. **"trace-specific 才行 vs 任意信号都行"** — **Paper 1 没做严格 ablation**！应该测"加一维随机噪声"或"加一维完全不相关的信号"作为对照，看是不是**只有 trace 这种 action-derived 信号有效**。**这是 Paper 1 里可以补的洞**。
3. **"这个 self 表征是稳定的/functional 的"** — 需要 exp4b 的 self-maintenance ablation 才能证。
4. **"跨越 gap 后能做什么"** — exp4 只证 probe 能读，没证这个表征能被下游任务利用（symbol grounding 是另一个 metric）。

### 3.5 常见误读

**误读 1**："trace 是 label shortcut"
- **错**。trace = |a| 的 EMA，**只携带 action 幅度信息**，不携带 is_trailing 标签。**系统需要学习**如何从 τ 的时序模式里区分 trailing 和 quiet。

**误读 2**："加一维随机噪声也能提升 recall"
- **可能对，但 Paper 1 没测**。这是**严格论证的一个洞**——如果任意一维输入都能提升，"trace 是最小充分"这个 claim 就弱化。**exp7+ 的一个自然方向就是补这个 ablation**。

**误读 3**："exp4 的 recall 只有 60%+，说明 h 里的 self 表征还很弱"
- **半对半错**。60% recall 在**高度类别不平衡**（trailing 只占 6%）下已经**显著超越 chance**，且相对 exp3 的 12% 是**5 倍提升**——**质变，不是量变**。**60% 而不是 100%**说明还有改进空间，但**足以证明"可读表征存在"**。

**误读 4**："exp4 是加了 label 让 aux 学习"
- **错**。aux head 在 exp3 里也存在，也在推 h 编码 trailing 标签。exp3 和 exp4 **唯一差别在 trace**，不在 aux 训练。**是架构变化（多一维输入）而非监督变化让 recall 突破**。

---

## 4. 合读：Encoding Gap 作为核心概念

### 4.1 为什么必须合读

**exp3 单独看**：negative result（读不出）——**看起来悲观、常识、无趣**
- ML 视角：分布式表征里线性 probe 读不出，是常识
- 如果没有 exp4，读者会问"so what？"

**exp4 单独看**：加信号就能读——**看起来平凡**
- 直觉：加更多信息，模型效果好，正常
- 如果没有 exp3 对比，读者会问"who cares？"

**合起来看**：**只差一维输入，recall 从 12% → 60%+**——**5 倍质变**
- **同一套模型、同一套训练、同一个 aux 施压、同一个 probe**
- **只差 GRU 输入 4d → 5d**
- **架构决定表征能否存在**

**这个对照才是 Paper 1 的原创贡献**——**不是"发现 encoding gap"**（gap 本身是分布式表征的常识），**是"证明跨越 gap 的最小充分构造是 1D 内源信号"**。

### 4.2 Encoding Gap 的精确 claim

**用 Paper 1 一句话表述**：**"causal utilization ≠ readable self-representation"**（因果利用 ≠ 可读自我表征）

**用我们讨论出的精确语言**：

> **系统能通过隐式 causal loop 做好预测（能用），不等于 h 里有一个可读的、独立的"我在动"维度（能读）。**
>
> **能用**：pred 通过分布式 h 里的信息综合出正确预测——**线性 readout 从多个 h 维度组合出结果**
>
> **能读**：h 里有一个特定方向对应 self 状态——**线性 probe 一个维度就能提取**
>
> **能用**的信息可以**弥散在 h 里**（分布式）；**能读**要求信息**集中到一个 readable 方向**（可指认）。**两者是不同的信息组织形式**。

### 4.2a **Canonical 三段总结**（比 Paper 1 §3.4 自己的表述更严谨）

**Evan 在 2026-07 讨论中提出的凝练版本**——**这是 exp3+exp4 到底证明了什么的最精确表述**：

**积极结论**：

> **Exp3**：没有编码通道，再多监督也学不到不存在的信息。
> **Exp4**：有了编码通道，动作历史能够进入内部表示，并被后续网络利用。

**否定 claim**（Paper 1 §7 limitation 承认但表述较弱的地方）：

> 它并没有直接证明：**Hidden 里形成了抽象的"自我（self）"概念**。

**统一表述**（原版）：

> **内部表征不能创造未曾编码的信息；自体痕迹首先是一种编码通道，然后才可能发展为更高级的自我表征。**

**统一表述**（严格化版，加 3 处限定词以关闭 ML 读者的技术追问）：

> **内部表征不能创造未以可读形式编码的信息；自体痕迹首先是一种持久化编码通道，然后才可能发展为更高级的自我表征。**

### 4.2b 为什么这个表述值得成为 canonical

**三个关键 shift**：

1. **消极结论精确化**：从"gap 存在"（模糊的负面结果）→ "未编码信息无法被监督重建"（明确的理论 claim）
2. **积极结论定位到正确层级**：**trace 首先是"编码通道"，不是"表征"**——这是 Paper 1 §3.4 表述可能被误读为"trace 引起自我涌现"的关键校正
3. **"可能"这个 qualifier 是精髓**：**不 claim 现在已经发展成自我表征**，只 claim 现在有了发展的**架构前提**。**这个谦逊使论述 defensible against reviewer skepticism**

**对比 Paper 1 §3.4 的表述**：
- **Paper 1 用 "self-representation"** 这种强术语（暗示"自我概念"已形成）
- **本表述用 "编码通道 → 可能发展为自我表征"**（明确区分"通道"和"表征"，且不 overclaim）

**这个校正**：**Paper 1 的贡献严格化为提供"跨越 encoding gap 的最小架构前提"，不是"引起自我涌现"**——**从 oversell 拉回到严格的、可辩护的 claim**。

### 4.3 Paper 1 核心贡献的分层

| 层 | 内容 | Novelty |
|---|-----|:------:|
| encoding gap 存在 | 分布式表征里线性 probe 读不出——ML 常识（Hinton 1986 起）| 低 |
| 具体化到 self-representation | 把 gap 应用到"我在动"这个概念，跨学科桥接 | 中 |
| **最小充分构造** | **1D EMA of \|a\| 就够** | **高**（原创）|
| **bio 对应** | trace ≈ proprioception（本体感觉）| **高**（跨学科连接）|
| 构造性 + 量化 + 发育框架 | 4 conditions + 12 falsified alternatives + 发育顺序 | **高**（方法论）|

**Paper 1 §6.2 说 "encoding gap is perhaps the most important finding for theories of self-awareness"**——**这句话对 ML 受众看是 oversell**，但**对认知科学/哲学受众看是真实的桥接贡献**。

**审稿人质疑"encoding gap 有什么新"，正确回答**：**"gap 本身不新，最小充分构造和跨学科桥接是新的"**。

---

## 5. 未解决的问题（open questions）

### 5.1 h 层的 A vs B 严格区分

**exp3/exp4 只用线性 probe 测 readability**——但 A vs B 严格来说是**h 内部结构**的问题：
- **A（表征分解）**：h 里有独立的 self 编码维度
- **B（相关变量）**：h 里有和 self 相关的信号，但不是独立维度

**线性 probe 通过对 A 有效，对 B 无效**——**但如果 probe 通过，我们区分不了 A 是"真的独立"还是"高度相关"**。

**需要 h 层的 counterfactual probe**（做 do 操作干预 h 特定维度，看下游预测响应）——**Paper 1 没做，是未来方向**。

### 5.2 trace 的最小性和 specificity ablation

**Paper 1 声称 1D trace 是最小充分**，但没做：
- **"加一维随机噪声"** 作对照
- **"加一维不相关信号"** 作对照
- **"加两维 trace"** 看是否更好

**如果任意一维都能提升，"trace 是最小充分"就弱化**。**这是一个可以补的 ablation**。

### 5.3 更长训练 / 更复杂 probe

**exp3 只测**：
- **线性 probe**（Linear(192, 1)）
- **80K 步训练**

**如果加**：
- **非线性 probe**（如 GRU 或 MLP probe）
- **更长训练**

**recall 会不会突破 12%？** 如果会，"encoding gap"就变成"线性可读性缺陷"，而不是"表征本身不存在"。

**Paper 1 有一个实验数据点** — 见 Section 3.3 Table 1 里的 "GRU probe (complex): 1.4%"——**更复杂的 probe 甚至更差**。这**加强了**"信息真的不在 h 里"这个 claim，但仍然是 empirical，不是理论证明。

### 5.4 Level 3 → 4：递归自我模型

**Paper 1 达到 Level 3（可读的 self 表征）后自然的下一步**：**Level 4——recursive self-model（"我知道我在动"的元表征）**。

**exp7 (metacog_level4) 是尝试**（`experiments_ext/metacog_level4/`）——目前 paused。

---

## 6. 附录：CET 视角的重解读

### 6.1 encoding gap = §13.8 第二条件缺失

**CET §13.8 的两条件命题**：
1. 理论必要性：$I(X; S_{t+1} \mid \text{已有条件集}) > 0$
2. 架构必要性：信息通路存在

**exp3 的失败 = 第二条件缺失**：
- 有理论条件（$M_t$ 携带独立预测信息）
- **没有架构通路**（obs 4d 里没有 M_t 的输入通道）
- → 优化压力存在（aux head 使劲推），但无处作用 → recall 卡在 12%

**exp4 的成功 = 打开架构通路**：
- 加一维 τ = 打开 M_t 的最小信息通道
- 现在两条件都满足 → recall 跳到 60%+

**这个理解让 encoding gap 从"神秘发现"变成"CET 理论预测"**——不神秘，是可预测的现象。

### 6.2 encoding gap = 分布式 vs 显式的 A vs B 区分

**用 CET §9.8 的两把尺**：
- **$D_{KL}^{obs}$ 吸收**：模型通过被动观察吸收约束的相关侧面
- **$D_{KL}^{int}$ 吸收**：模型通过主动干预吸收约束的因果侧面

**exp3 的**：h 内部至多是 $D_{KL}^{obs}$ 层的吸收（弥散分布式），**不是** $D_{KL}^{int}$ 层的独立可分解

**exp4 加 trace 让 h 里出现某种更结构化的表征**——但**Recovery/Probe 都是 obs 层测试**，**没能在 h 层做 $D_{KL}^{int}$**（干预 h 特定维度看下游响应）——**A vs B 严格区分仍需 exp7+**。

### 6.3 Encoding gap 在三层框架里的位置

**用 papers/three_levels/ 的三层框架**：

- **L1 Dynamics**：GRU 提供 h 轨迹
- **L2 Readout**：pred 学线性投影
- **加 aux 后**：局部 L3 shaping（aux 梯度回流塑造 h）

**exp3 里 L3 shaping 存在但不够** —— **架构限制**（4d 输入没有 M_t 通道）**约束了 L3 能塑造的方向**

**exp4 加 τ 扩展了输入维度** —— **给 L3 shaping 一个新方向可以塑造**

**encoding gap 的三层框架视角**：**L3 shaping 的塑造能力 = f(架构输入通道)**——**再强的 shaping 也不能创造输入维度里不存在的信息**。

---

## 学习检查清单

读完这份文档后，你应该能回答以下问题：

- [ ] exp2 引出的问题是什么？为什么需要 exp3 来回答？
- [ ] exp3 里 action 为什么必须是外部随机？换成 W_action(h) 会怎样？
- [ ] Aux head 和 Probe 的分工是什么？两者都读 h，区别在哪？
- [ ] BurstGate 的三个类别（active/trailing/quiet）为什么 trailing 是关键测试？
- [ ] exp3 的 12% recall 意味着什么？如果换非线性 probe 会不同吗？
- [ ] exp4 和 exp3 的架构差别只有什么？为什么这个"最小对照"重要？
- [ ] trace 的三个关键性质（1D、内源、衰减）各自的作用？
- [ ] 为什么必须合读 exp3 + exp4？单独看各有什么局限？
- [ ] encoding gap 的精确 claim 是什么？为什么"能用 ≠ 能读"？
- [ ] Paper 1 的核心贡献不是"发现 gap"，是什么？
- [ ] exp3/exp4 后 open questions 有哪些？exp7+ 的方向是什么？

---

**下一步（读完这份文档后可选路径）**：

1. **exp4b 单独深读**（Self-Maintenance）—— 补上 exp2 Layer 3 引用的机制细节
2. **exp5 深读**（Async Awakening）—— 时序作为 identifying assumption
3. **exp6 深读**（Agency Gain）—— 定量化 + Pearl Layer 3 反事实
4. **回过头看 Paper 1 §6.2**——用现在的理解重新读原文的 discussion，找到还有哪些表述可以精细化
5. **跑 exp3/exp4 empirical 验证**——用 `--quick` mode 亲自跑一遍看数字
