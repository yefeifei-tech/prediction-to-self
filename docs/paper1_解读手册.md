# Paper 1 中文解读手册

**Paper:** *From Prediction to Self: Developmental Conditions for Agency in Minimal Neural Systems* ([arXiv:2606.05605](https://arxiv.org/abs/2606.05605))

**本手册用途**：帮助自己（作者）快速回忆 Paper 1 的实验结构、每个指标在证据链中的位置、常见误读、以及代码定位。**不是给读者的对外说明**，是内部参考。

---

## 目录

1. [一句话概括](#一句话概括)
2. [Self 是什么：操作性四联定义](#self-是什么操作性四联定义)
3. [核心命题与四条件](#核心命题与四条件)
4. [方法论核心：干预式因果推断](#方法论核心干预式因果推断)
5. [八步证据链（完整版）](#八步证据链完整版)
6. [分实验详解](#分实验详解)
7. [五指标 vs 八指标：为什么不能压缩](#五指标-vs-八指标为什么不能压缩)
8. [关键数字速查表](#关键数字速查表)
9. [常见误读与澄清](#常见误读与澄清)
10. [代码结构导航](#代码结构导航)

---

## 一句话概括

**Paper 1 用一组递进的干预和对照，把 "self" 作为一个概念从统计相关中一层层剥离出来**——每一步都排除一个替代解释，最终把 self 定义为 "因果闭环 + 显式表征 + 自我维持 + 发育时序" 的合取。

---

## Self 是什么：操作性四联定义

这是最容易被误读的地方。**Paper 1 没给 self 一个本体论定义（"self 是 X"），它给的是操作性定义（"当以下条件满足时，我们说系统有 self-representation"）**。这个区别很重要——它避开了所有形而上学承诺，只谈可测量的属性。

### Self 不是什么

- ❌ **不是 h 里的某个特定位置或神经元**：没有 "self neuron"，没有 "self subspace"
- ❌ **不是一个可以指认的实体**：paper 避免所有关于"意识"、"感受质"的语言
- ❌ **不是一个符号或 label**：symbol grounding 是后续 paper 的内容，Paper 1 里没有
- ❌ **不是 agent 的行为能力**：仅"能作用于世界"不够（exp2 已经有 action loop，但 exp3 显示 h 里没有 self）

### Self 是什么：四联操作定义

系统拥有 self-representation，**当且仅当**以下四个条件同时满足：

#### (1) 因果层面：Agency Gain > 0

$$A = \text{Err}_B - \text{Err}_A > 0$$

即："知道我做了什么"能够超越"只知道世界状态"来改善预测。
- `Err_B`：只用 h 预测（不知道 action）
- `Err_A`：用 h + action 预测

$A > 0$ 是 self 存在的**量化标志**——它意味着 action 作为一个变量携带了 h 之外的独立预测信息。

#### (2) 表征层面：h 里有可读的"我在动作"维度

detached probe 从 h 读 trailing 状态：**recall ≥ 60%**（exp4 达成）。

关键洞察：**因果利用 ≠ 自我表征**。exp3 里 pred_A 能靠权重矩阵完美补偿 action 影响（因果利用充分），但 h 里没有"我在动"这个可读维度（自我表征缺失）。**self 不是"能做什么"，是"能读出'我在做'"**。

#### (3) 动力学层面：自我维持

撤走 aux 监督后，h 里的 self 编码**保持** ≥ 90%（exp4b causal 组 94.9%）。

含义：self 必须**被系统的动力学内化**，而不是被外部损失函数逼着表达。**只有当 self 对系统本身有用（因果闭环上必需），它才自我维持**。

#### (4) 发育层面：正确的时序涌现

必须满足：`persistent state → causal loop → trace → async awakening` 的严格顺序（exp5）。

含义：**self 是一个发育产物**，不是可以直接安装的模块。顺序错了（同步训练感知和动作），四组件都在也不成立。

### 一句话回答"self 是什么"

**Self 是 h 相对于世界的一种因果分解模式**——具体来说，是 h 中那个能够可靠区分"这个 obs 变化是我做的" vs "这个变化是世界发生的"、并且这种区分是自我维持的、通过特定发育顺序涌现出来的功能结构。

**类比**：self 不是 h 里的一个"部件"，而是 h 的一种"姿态"（stance）——就像"平衡"不是身体里的某个器官，而是骨骼-肌肉-前庭系统之间的一种动态关系。

### 数学化的定义（论文正式版）

系统 $\mathcal{S}$ 拥有 self-representation，当且仅当存在观察通道 $c$ 和内部状态 $h$，使得：

1. **因果**：$H(\text{obs}_c^{t+1} \mid h^t) - H(\text{obs}_c^{t+1} \mid h^t, a^t) > 0$
2. **可读**：$\exists\, \text{linear probe}\ \pi: I(\pi(h^t);\ \mathbb{1}[a^{t-k}\neq 0]) > \tau$ （$k > 0$，如 50）
3. **持续**：撤去监督后条件 (2) 保持 $> 90\%$
4. **发育**：以上三个性质通过 $\{c_1 \prec c_2 \prec c_3 \prec c_4\}$ 的时序涌现，顺序不可交换

**self 就是这个联合条件所刻画的功能结构**——不多不少。

### 与哲学传统的关联

Paper 1 站在 **enactivism（生成认知）+ predictive processing（预测加工）** 的交汇处：

- **Predictive Processing (Friston)**：self 由预测动力学定义，$A = \text{Err}_B - \text{Err}_A$ 直接对应 free energy 视角下的 "self-model utility"
- **Enactivism (Varela, Thompson)**：self 不是静态实体，而是通过和世界的因果耦合 **enact**（生成）出来的
- **Metzinger's Self-Model Theory**：h 里的 self-representation 就是 Metzinger 意义上的 self-model；encoding gap 是他讲的 "phenomenal transparency" 的可测量对应

Paper 1 的**新贡献**：把这些哲学立场变成了**可测量的操作定义 + 可验证的构造性证明**。读者不用相信任何形而上学立场，只需接受这四个操作条件——然后 Paper 1 证明满足这些条件的最小系统是可构造的。

### 被人问 "self 到底是啥" 的标准答复

> Self 在 Paper 1 里是一个**四联操作性构念**：因果分解（Err_B − Err_A）+ 可读表征（probe recall）+ 自我维持（ablation retention）+ 发育时序（curriculum order）。它不是 h 里的某个东西，而是 h 相对于世界的一种功能关系。这个定义避开了本体论承诺，只谈可测量的属性——但同时给出了构造性证明：满足这四个条件的最小系统是什么样子。

---

## 核心命题与四条件

**核心问题**：一个最小神经系统（192 维 GRU），从"预测世界"发展到"能区分'我做的'和'世界发生的'"，需要哪些充分条件？必须按什么顺序满足？

**答案：四个条件，严格有序。** 任何一个缺席或顺序颠倒，都不成立。

```
Level 0（预测）
   │
   ▼
条件 1: Persistent state
   │   稳定的 attractor（GRU + multi-scale EMA）
   │   ↳ 若 h 不稳定，后续所有解读无效
   │
   ▼
条件 2: Causal action loop
   │   action = f(h), obs[0] += GAMMA·action
   │   ↳ 让 self 有"作用于世界"的通路
   │
   ▼
条件 3: Proprioceptive feedback (trace τ_t)
   │   把 |action| 的 EMA 喂回 GRU
   │   ↳ 让 h 有通道保留"我在动作"
   │
   ▼
条件 4: Asynchronous awakening
   │   感知先训练稳定，动作再解冻
   │   ↳ 时序不能颠倒，同步训练会互相污染
   │
   ▼
Level 3+（可显式区分 self vs world）
```

---

## 方法论核心：干预式因果推断

**Paper 1 骨子里是因果推断，只是没用 SCM/DAG 的语言明说。**

对应 Pearl 的三层因果阶梯：

| Pearl 层次 | Paper 1 里的操作 | 出处 |
|-----------|-----------------|------|
| **Layer 1 观察** | AG = Err_B − Err_A（量化标尺） | [exp6](../experiments/exp6_measurement.py) |
| **Layer 2 干预** | 断开 world 端的 action→obs 通路 | [exp2 spike test](../experiments/exp2_causal.py#L143) |
| **Layer 3 反事实** | 喂给 pred_A 假的 action（0 或 −a） | [exp6 counterfactual](../experiments/exp6_measurement.py#L229) |
| **混淆控制** | AR(1) Control 组（匹配统计） | [exp2](../experiments/exp2_causal.py#L100), [exp4b](../experiments/exp4b_self_maintenance.py) |

**重要**：Paper 1 **已经**用了 Layer 3 反事实，不只是 Layer 2 干预。exp6 的 zero-action / wrong-action 就是标准反事实查询。

---

## 八步证据链（完整版）

从"有关联"到"是自我"的完整证据链：

```
0. 构造（不是证明，是设定）
   obs[0] += GAMMA * action   ← 实验者植入的因果通路
   ↓
1. AG > 0（观察层，量化标尺）
   pred_A 拿 action，pred_B 不拿
   AG = Err_B − Err_A > 0
   ↳ 排除：无——这是构造的必然，AG 只是刻度
   ↓
2. [Pearl Layer 2] exp2 world-side disconnect
   切断 obs[0] += GAMMA·a
   → 只有 ch0 spike > 2x，其他通道 ~1x
   ↳ 排除："action 影响所有通道"
   ↓
3. [混淆控制] exp2 Control (AR(1))
   匹配自相关 φ=0.95、方差、幅度
   唯一差异：action = f(h) vs 外部 AR(1) 噪声
   → Causal long-disconnect 后能 recovery，Control 不能
   ↳ 排除："任何统计匹配的信号都能产生这个效果"
   ↓
4. [编码分离] exp3 vs exp4 encoding gap
   同一 setup，只差 GRU 输入维度（4d vs 5d）
   → trailing recall 12% vs 60%+
   ↳ 排除："因果利用就等于自我表征"
   ↳ 概念发现：causal utilization ≠ readable self-representation
   ↓
5. [Pearl Layer 3] exp6 counterfactual
   给 pred_A 喂 action=0 或 −action
   → err_a_zero、err_a_wrong 显著飙升
   ↳ 排除："pred_A 只是把 action 当装饰品"
   ↓
6. [功能验证] exp4b self-maintenance ablation
   撤走 aux 监督
   → Causal 保持 94.9%，Control 塌到 53.9%
   ↳ 排除："自我表征只是被外部监督逼出来的假象"
   ↓
7. [时序必要性] exp5 async vs sync
   FAST/MEDIUM/SLOW 同步基线全部失败
   Async（P1 冻结 → P2a 巩固 → P2b 解冻）成功
   ↳ 排除："只要架构齐全就够了，训练顺序无关"
   ↳ 时序是 identifying assumption，不是超参
   ↓
8. [定量测量] exp6 Agency Gain 完整测试
   多策略 × spike test × Lorenz 混沌 × 延迟鲁棒性
   ↳ 综合评分卡：多个正交指标交叉验证
```

**每一步排除一个替代解释。少一步，就有替代解释没被排除，self 这个概念就没被钉死。**

---

## 分实验详解

### exp1: Perception — 稳定 attractor

**文件**：[experiments/exp1_perception.py](../experiments/exp1_perception.py)

**问题**：连续预测的 GRU 能否形成稳定内部结构？

**为什么先做**：如果 h 是随机噪声或不断漂移，后面所有对 h 的解读都是自欺。这是**前提性检查**，不是核心结论。

**Setup**：4 通道正弦信号、GRU(192) + multi-scale EMA、单预测头、50K 步、扰动 + 新颖响应事件

**指标（7 项 scorecard，通过率 6/7 视为达标）**：
1. 有效维度 < 30%（低维吸引子）
2. 幂律衰减（多时间尺度）
3. 扰动恢复 > 50%（attractor 稳定）
4. 残差白噪声（预测充分提取结构）
5. 新颖响应 > 1.0（能对外部变化响应）
6. 谱分离（4 个 EMA 组有不同峰值频率）
7. 误差平稳（长期不发散）

**关键代码**：[exp1_perception.py:87](../experiments/exp1_perception.py#L87) `run_scorecard()`

---

### exp2: Causal Budding — 因果萌芽

**文件**：[experiments/exp2_causal.py](../experiments/exp2_causal.py)

**问题**：系统的 action 只改变 obs[0]，它能否学到"我影响的是哪个通道"？

**Setup**：
- **Causal 组**：`action = f(h) = W_action · h`，GAMMA=2.0
- **Control 组**：`action = AR(1) 噪声`（φ=0.95），统计匹配
- 单预测头（no dual heads yet），60K 步

**关键指标**：
- **通道特异 spike ratio**：断开 action，ch0 error 飙升，其他不变
- **Long-disconnect recovery**：断开 2000 步后 ch0 error 恢复程度

**期望结果**：
- Causal ch0 spike > 2x
- Causal recovery > Control recovery（关键对照）

**关键代码**：
- 训练：[exp2_causal.py:87](../experiments/exp2_causal.py#L87) `train_model()`
- Spike 测试：[exp2_causal.py:143](../experiments/exp2_causal.py#L143) `spike_test()`
- 长断连：[exp2_causal.py:202](../experiments/exp2_causal.py#L202) `long_disconnect_test()`

**在证据链中的位置**：**步骤 2 + 3**（Pearl Layer 2 干预 + 混淆控制）

---

### exp3: Encoding Gap — 编码鸿沟

**文件**：[experiments/exp3_encoding_gap.py](../experiments/exp3_encoding_gap.py)

**问题**：即使 aux head 强推 h 编码"我在动作"，只有 obs(4d) 输入的 GRU 能不能做到？

**Setup**（关键：**没有 trace**）：
- GRU 输入 = obs(4d) only
- Aux head 三分类（active / trailing / quiet），梯度打进 h
- **BinaryProbe 用 detached h** 测 trailing 时段能否读出

**BurstGate 机制**：随机开关 action bursts，产生 active/trailing/quiet 三个 phase。`trailing` 是 action 刚停后 50 步——测的是"h 能不能在 action 消失后还记得刚才动过"。

**期望结果**：**trailing recall ≈ 12%**（接近随机，encoding gap 存在）

**关键代码**：[exp3_encoding_gap.py:73](../experiments/exp3_encoding_gap.py#L73) `Model` (4d GRU), [exp3_encoding_gap.py:115](../experiments/exp3_encoding_gap.py#L115) `BinaryProbe`

**核心洞察**：pred_A 能预测好（隐式补偿），但 h 里没有"我在动"的可读维度。**这就把"prediction quality 好"和"self-representation 存在"在同一个模型里分离开了**。

**在证据链中的位置**：**步骤 4**（编码分离，与 exp4 对照）

---

### exp4: Proprioceptive Breakthrough — 本体感受突破

**文件**：[experiments/exp4_proprioception.py](../experiments/exp4_proprioception.py)

**问题**：加一维 trace（|action| 的 EMA）能否打破 encoding gap？

**Setup**：**与 exp3 完全相同，只差 GRU 输入 4d → 5d**（多了 trace_t）

trace 定义：`τ_t = 0.95 · τ_{t-1} + 0.05 · |a_t|`

**期望结果**：**trailing recall ≈ 60%+**（gap 被打破）

**关键代码**：[exp4_proprioception.py:70](../experiments/exp4_proprioception.py#L70) `ProprioceptiveModel` (5d GRU), [exp4_proprioception.py:87](../experiments/exp4_proprioception.py#L87) `update_trace()`

**核心洞察**：
- 这是**最小对照**——只加一维，其他全同
- 证明 encoding gap 是**架构决定的**，不是 loss 或数据决定的
- trace **是内源的**（完全从 action 自身派生，不用外部 label）——这是关键的 CET 属性

**在证据链中的位置**：**步骤 4**（与 exp3 对照）

---

### exp4b: Self-Maintenance — 自我维持

**文件**：[experiments/exp4b_self_maintenance.py](../experiments/exp4b_self_maintenance.py)

**问题**：exp4 的 60% recall 会不会只是 aux head 逼出来的假象？撤走 aux 会不会秒崩？

**Setup**：
- 训完 Phase 2 后加 **Phase 3: ablation**——移除 aux loss，继续跑 5000 步
- 对照 Causal（action from h）vs Control（AR(1)）

**期望结果**：
- Causal 保持 **94.9%**
- Control 塌到 **53.9%**（接近 chance）

**关键代码**：[exp4b_self_maintenance.py:1](../experiments/exp4b_self_maintenance.py) 起

**核心洞察**：**self-representation 只有在对系统本身有用时才自我维持**。Causal 组的 action 是内源的、self 表征参与 action 生成，撤 aux 后 GRU 动力学已经内化了这个信号；Control 组的 action 是外源的，self 编码对系统无用，aux 一撤就散。

**在证据链中的位置**：**步骤 6**（功能验证：因果有用 → 自我维持）

---

### exp5: Asynchronous Awakening — 异步唤醒

**文件**：[experiments/exp5_async_awakening.py](../experiments/exp5_async_awakening.py)

**问题**：感知和动作能同时学吗？还是必须顺序？

**Setup**：
- **Async**：P1 冻结 W_action 100K 步 → P2a 巩固 60K 步（LR 降到 1e-4）→ P2b 解冻 W_action 60K 步
- **Sync 三基线**：W_action 从第 0 步就解冻并训练
  - FAST（W_action LR=1e-3）：动作太强，污染感知
  - SLOW（1e-4）：动作太弱
  - MEDIUM（5e-4）：折中

**关键指标**（两个必须同时满足）：
- spike > 2.0（有 agency）
- trailing > 30%（有 awareness）

**期望结果**：只有 async 同时通过两个指标；三个 sync 基线都至少有一个失败。

**关键代码**：[exp5_async_awakening.py:98](../experiments/exp5_async_awakening.py#L98) `FullModel`

**核心洞察**："顺序"不是超参调不过去的技术问题，是**identifying assumption**。感知稳定必须先于动作学习，否则 self 无法涌现。

**在证据链中的位置**：**步骤 7**（时序必要性）

---

### exp6: Measurement — Agency Gain

**文件**：[experiments/exp6_measurement.py](../experiments/exp6_measurement.py)

**问题**：把整个发展过程变成连续、可比较的定量测量。

**Setup**：
- Dual heads：pred_A(h+action) 和 pred_B(h)
- Agency gain: **A = Err_B − Err_A**
- 支持四种 flag：`--quick --lorenz --trace --delay`
- 三种策略对照：Forward sampling, Direct AG gradient, Gradient disagreement

**关键指标**：
- **Prediction gap**：`(Err_B − Err_A) / Err_B × 100%`
- **Counterfactual spike test**：
  - `err_a_zero`：pred_A 拿到 action=0 的误差
  - `err_a_wrong`：pred_A 拿到 −action 的误差
  - `spike_zero = err_a_zero / err_a_normal`
- **Action autocorrelation**：action 时序自相关

**Scorecard**（4 项）：
1. gap > 80%
2. spike > 1.5x
3. autocorr > 0.5
4. Phase 1 spike < Phase 2b spike

**关键代码**：
- Agency gain 训练：[exp6_measurement.py:106](../experiments/exp6_measurement.py#L106) `run_full_training()`
- **反事实 spike test**：[exp6_measurement.py:229](../experiments/exp6_measurement.py#L229) `run_spike_test()`

**核心洞察**：
- gap 提供连续标尺，测发展进度
- spike test 提供 counterfactual 证据，排除"pred_A 只是把 action 当装饰"
- Lorenz + delay 测**鲁棒性**：agency 不是只在特定信号或即时反馈下才成立

**在证据链中的位置**：**步骤 1**（AG 观察）+ **步骤 5**（Layer 3 反事实）+ **步骤 8**（综合测量）

---

## 五指标 vs 八指标：为什么不能压缩

我最初尝试过用 "五指标" 概括，但 **Paper 1 的证据链其实是 8 步的**，压缩到 5 会丢掉两个关键环节：

### 压缩会丢失的两个环节

**A. exp3 encoding gap 作为独立发现**
- "五指标" 版本把 probe 归为"读取指标"
- 但 exp3 → exp4 的**对比本身**是概念性发现：`causal use ≠ self-representation`
- 这不是"能不能读出"的问题，是"因果利用和自我表征是两回事"的问题
- **压缩会把它降级为一个技术指标，丢掉概念创新**

**B. exp5 async awakening 作为时序条件**
- "五指标" 版本是"逐层加东西"
- 但 Paper 1 的完整论证还包括"**顺序不能乱**"
- 这不是加变量，是加时序约束这个 identifying assumption
- **压缩会漏掉"发育顺序"这个第 4 条件**

### 五指标版本的其他不准确

如果你之前听过或看到"五指标"版本，注意这些容易搞混的地方：

| 五指标版本的说法 | 实际情况 |
|----------------|---------|
| "Spike test 是 Pearl 第 2 层" | Paper 1 有**两个** spike test：exp2 世界端断开（Layer 2），exp6 pred_A 反事实（**Layer 3**） |
| "Control 组的 AG 低" | Paper 1 **没有**跑 Causal vs Control 的 AG 对比。Control 的作用是 long-disconnect recovery（exp2）和 self-maintenance retention（exp4b） |
| "Probe 70% → 65.4%" | Paper 1 数字是 **12% → 60%+**。70%/65.4% 属于 v10.3 或后续 paper |
| "Symbol grounding 是第 5 步" | Paper 1 **没有** symbol grounding。那是 v10.3 / Paper 2 的内容 |

---

## 关键数字速查表

| 指标 | 数值 | 出处 |
|------|-----|------|
| exp2 ch0 spike ratio (Causal) | > 2.0x | exp2 spike test |
| exp3 trailing recall（encoding gap） | ≈ **12%** | exp3 probe |
| exp4 trailing recall（breakthrough） | ≈ **60%+** | exp4 probe |
| exp4b retention (Causal) | **94.9%** | exp4b ablation |
| exp4b retention (Control) | **53.9%** | exp4b ablation |
| exp6 pred gap | > 80% | exp6 scorecard |
| exp6 spike zero | > 1.5x | exp6 counterfactual |
| exp5 async spike | > 2.0 | exp5 |
| exp5 async trailing | > 30% | exp5 |

---

## 常见误读与澄清

### 误读 0："self 是 h 里的某个东西"（最常见）

**错**。Paper 1 采用**操作性定义**，不是本体论定义。self 不是 h 里的某个位置、某组神经元、或某个可以指认的实体，而是 h **相对于世界**的一种功能关系——由四个可测量条件（Agency Gain > 0 + probe recall + ablation retention + 发育时序）联合刻画。详见 [Self 是什么：操作性四联定义](#self-是什么操作性四联定义)。

被问到"self 到底在 h 的哪里"，正确回答是：**"这个问题问错了。self 不占据 h 的某个位置，它是 h 被使用的一种模式。"**

### 误读 1："Paper 1 是关于 prediction 的"

**不是**。Paper 1 是关于**从 prediction 涌现出 self**。prediction 只是起点条件（exp1），核心贡献在 exp3-exp5：encoding gap、self-maintenance、async awakening。

### 误读 2："Agency gain 证明了系统有 agency"

**AG > 0 是构造上必然的**（pred_A 拿 action、pred_B 不拿）。AG 的作用是**量化标尺**，不是因果证据。真正证明 agency 的是 spike test（Pearl 层次）+ Control 组（混淆控制）+ encoding gap（隐式/显式分离）。

### 误读 3："只要给系统 action loop 它就有 self 表征"

**错，这是 encoding gap 揭示的核心**：exp2 已经有 action loop，系统能**隐式**利用因果（pred_A 权重补偿），但 h 里**没有**显式的"我在动作"表征。需要 trace（本体感受）才能跨过这道坎。

### 误读 4："aux head 让 h 编码 self，问题解决了"

**错**。exp3 就有 aux head，但 GRU 输入只有 obs(4d) 时 recall 只 12%。**架构决定表征能否存在，loss 只是驱动力**。aux head 无法凭空创造 h 里不存在的通道。

### 误读 5："Control 组是为了对比性能"

**错，Control 组不是性能对比**。Control 是**混淆变量控制**——匹配 action 的统计分布，唯一变化"action 是否来自 h"。这样剩下的差异必然归因于"因果闭环"这个变量。这是 Pearl 意义上的干预实验设计。

### 误读 6："Paper 1 停留在 Pearl 第 2 层"

**错，Paper 1 触到第 3 层**。exp6 的 counterfactual spike test（喂 pred_A 假的 action）是标准的 Layer 3 反事实查询。v10.3 是把这套方法**系统化**，不是**发明**。

### 误读 7："Symbol grounding 是 Paper 1 的一部分"

**不是**。Paper 1 没有 symbol grounding 实验。任何提到 "'I' 符号"、"BA 80.1%"、"generalization 83.8%" 的都属于后续工作（v10.3 或 Paper 2）。

---

## 代码结构导航

```
prediction-to-self/
├── train.py                          # 训练入口
├── core/
│   ├── model.py                      # AgencyModel（GRU + EMA + dual heads + W_action）
│   ├── world.py                      # SineSignal + LorenzSignal
│   └── lorenz.py                     # Lorenz attractor
├── experiments/
│   ├── exp1_perception.py            # §3.1 稳定 attractor
│   ├── exp2_causal.py                # §3.2 因果萌芽（含 Control 组）
│   ├── exp3_encoding_gap.py          # §3.3 编码鸿沟（12% recall）
│   ├── exp4_proprioception.py        # §3.4 本体感受突破（60%+ recall）
│   ├── exp4b_self_maintenance.py     # §3.2 自我维持消融（94.9% vs 53.9%）
│   ├── exp5_async_awakening.py       # §3.5 异步唤醒
│   └── exp6_measurement.py           # §3.6 agency gain + counterfactual
└── figures/
    └── gen_fig1_attractor.py ... gen_fig6_strategy.py
```

**关键类速查**：

| 类 | 位置 | 作用 |
|---|-----|-----|
| `AgencyModel` | [core/model.py:20](../core/model.py#L20) | 主模型：GRU + EMA + dual heads + W_action |
| `PerceptionModel` | [exp1_perception.py:52](../experiments/exp1_perception.py#L52) | 简化版（无 action、单头） |
| `SingleHeadModel` | [exp2_causal.py:50](../experiments/exp2_causal.py#L50) | 有 action 但单头（测隐式分解） |
| `Model` (4d) | [exp3_encoding_gap.py:73](../experiments/exp3_encoding_gap.py#L73) | encoding gap 用（无 trace） |
| `ProprioceptiveModel` (5d) | [exp4_proprioception.py:70](../experiments/exp4_proprioception.py#L70) | breakthrough 用（加 trace） |
| `BinaryProbe` | 多处 | 从 detached h 测可读性 |
| `AuxHead` | 多处 | 从 h 分类（梯度打进 h） |
| `BurstGate` | 多处 | 生成 active/trailing/quiet phase |

**运行**：
```bash
python -m experiments.exp1_perception
python -m experiments.exp2_causal --quick
python -m experiments.exp6_measurement --trace
python -m experiments.exp6_measurement --lorenz --trace --delay
```

---

## 附：CET 视角的一致性检查

Paper 1 是 CET 的 Paper 1，可以用 CET §13.8 的**条件集扩张逻辑**验证发育四条件是否满足：

对每一层 X → S_{t+1} 的跳跃：
1. **信息价值**：`I(X; S_{t+1} | 已有条件集) > 0`
2. **架构通路**：X 必须有真实的信息通路到达 h

| 条件 | 变量 | 架构通路 | 验证实验 |
|------|-----|---------|---------|
| 1 | Persistent state (h_multi) | GRU + EMA | exp1 scorecard |
| 2 | Causal action loop | W_action(h) → obs[0] | exp2 recovery gap |
| 3 | Proprioceptive trace (τ_t) | GRU 输入第 5 维 | exp3 vs exp4 recall gap |
| 4 | Async time-ordering | 冻结/解冻 W_action 的 curriculum | exp5 sync vs async |

**每一条都是"信息价值 + 架构通路"双条件同时满足**。少一个都不成立——这正是 Paper 1 的构造性论证。
