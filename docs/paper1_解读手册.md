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
6. [训练模式对照：哪些权重在动？](#训练模式对照哪些权重在动)
7. [分实验详解](#分实验详解)
8. [证据三元组速查表](#证据三元组速查表)
9. [五指标 vs 八指标：为什么不能压缩](#五指标-vs-八指标为什么不能压缩)
10. [关键数字速查表](#关键数字速查表)
11. [常见误读与澄清](#常见误读与澄清)
12. [代码结构导航](#代码结构导航)
13. [用 CET 重读 Paper 1](#用-cet-重读-paper-1)

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

**A = Err_B − Err_A > 0**

即："知道我做了什么"能够超越"只知道世界状态"来改善预测。
- `Err_B`：只用 h 预测（不知道 action）
- `Err_A`：用 h + action 预测

A > 0 是 self 存在的**量化标志**——它意味着 action 作为一个变量携带了 h 之外的独立预测信息。

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

系统 S 拥有 self-representation，当且仅当存在观察通道 c 和内部状态 h，使得：

1. **因果**：H(obs_c^{t+1} | h^t) − H(obs_c^{t+1} | h^t, a^t) > 0
2. **可读**：∃ 线性 probe π，使 I(π(h^t); 𝟙[a^{t−k} ≠ 0]) > τ　（k > 0，如 50）
3. **持续**：撤去监督后条件 (2) 保持 > 90%
4. **发育**：以上三个性质通过 {c₁ ≺ c₂ ≺ c₃ ≺ c₄} 的时序涌现，顺序不可交换

**self 就是这个联合条件所刻画的功能结构**——不多不少。

### 与哲学传统的关联

Paper 1 站在 **enactivism（生成认知）+ predictive processing（预测加工）** 的交汇处：

- **Predictive Processing (Friston)**：self 由预测动力学定义，A = Err_B − Err_A 直接对应 free energy 视角下的 "self-model utility"
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

| Pearl 层次 | 什么类型的操作 | Paper 1 里的对应 |
|-----------|--------------|-----------------|
| **Layer 1 观察** `P(Y\|X)` | 只看数据，不干预 | AG = Err_B − Err_A（正常运行下测量） |
| **Layer 2 干预** `P(Y\|do(X))` | 强制改变某个变量 | exp2 spike test、recovery test、exp4b ablation 都是 |
| **Layer 3 反事实** `P(Y_x\|X=x', Y=y')` | 给定实际发生的 h, obs，问"如果 action 当时是别的会怎样" | exp6 counterfactual spike（喂 pred_A 假 action） |
| **混淆控制** | 匹配统计变量以排除关联性替代解释 | AR(1) Control 组（[exp2](../experiments/exp2_causal.py#L100), [exp4b](../experiments/exp4b_self_maintenance.py)） |

**重要**：Paper 1 **已经**用了 Layer 3 反事实，不只是 Layer 2 干预。exp6 的 zero-action / wrong-action 就是标准反事实查询。

### Pearl 分层 ≠ 证据链分层（避免我曾经的混淆）

Paper 1 里有**两个不同的 "1/2/3 分层"**，一定要**分清楚**：

| 分层 | 关于什么 | Level 1 | Level 2 | Level 3 |
|-----|---------|---------|---------|---------|
| **Pearl 因果阶梯** | **查询类型**（用了什么操作） | 观察 | 干预 | 反事实 |
| **证据链分层**（本手册的表述） | **claim 类型**（建立了什么结论） | 存在性 | 性质 | 功能 |

**这两个分层是正交的两个轴**。举例说明：

- **Spike test** 是 **Pearl Layer 2 干预**（切断 action→obs 通路），但它建立的 claim 是**证据链 Level 1 存在性**（"系统学到了通道特异的模型"）
- **Recovery test** 也是 **Pearl Layer 2 干预**（长期切断），但它建立的是**证据链 Level 2 性质**（"是因果结构而非统计关联"）
- **Self-maintenance** 也是 Pearl Layer 2（撤走 aux loss，元层干预），建立的是**证据链 Level 3 功能**（"表征功能性内化"）
- **exp6 counterfactual** 是 **Pearl Layer 3 反事实**，是唯一真正在 Pearl 意义上升到第 3 层的

**换句话说**：**Paper 1 的三个证据层，都用相同的 Pearl 层次（Layer 2 干预）实现**——它们的区别不在**用了什么因果操作**，而在**测出了什么 claim**。

**为什么只有 exp6 够得着 Pearl Layer 3？这是架构决定的**：反事实查询要求"固定 h 与世界、只篡改喂进预测器的 action 值"，因此预测头必须有一个**显式的 action 输入槽**可供篡改。exp5/exp6 的 `pred_A` 把 action 拼成第 193 维输入（`Linear(192+1, 4)`），正好提供了这个槽；而 exp2、exp4b 的单头 `pred(h)`（`Linear(192, 4)`）根本没有 action 输入维，只能从**世界端**干预（断 `obs[0] += γ·a`）——所以单头实验在架构上就封顶 Pearl Layer 2。**"能问哪一层因果"是被预测头维度写死的。**

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
2. [Pearl Layer 2 · 通道特异性] exp2 world-side disconnect (spike test)
   切断 obs[0] += GAMMA·a
   → Causal 组内 ch0 spike ≫ ch1, ch2, ch3
   → Control 组内 ch0 spike 也 ≫ 其他通道（两组都通过——这是 desired）
   ↳ 排除："系统学到了 nothing / 系统学到'世界很乱'但不定位 / action 影响所有通道"
   ↳ 建立 Level 1（存在性）：两组都学到了"action 影响 ch0"这个通道特异模型
   ↓
3. [混淆控制 · 因果 vs 统计] exp2 Control (AR(1)) + long-disconnect recovery
   匹配自相关 φ=0.95、方差、幅度
   唯一差异：action = f(h) vs 外部 AR(1) 噪声
   → Causal long-disconnect 后能 recovery (~75%)，Control 不能 (~57%)
   ↳ 排除："两组学到的都是同类型模型"
   ↳ 建立 Level 2（性质）：只有 Causal 是结构化因果模型（可分解），Control 只是统计关联
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

## 训练模式对照：哪些权重在动？

**这是理解 Paper 1 架构最容易被忽略的关键细节**。不同实验里，**只有部分权重在训练**，其他权重保持随机初始化不动——这是刻意的设计。

### 总表

| 实验 | Phase | `pred` | `GRU` | `W_action` | `aux`/`probe` | 说明 |
|------|-------|:-----:|:-----:|:----------:|:-------------:|------|
| exp1 | — | ✓ | ✗ | 无 | 无 | Reservoir computing |
| **exp2** | — | **✓** | **✗ 冻结** | **✗ 冻结** | 无 | **Reservoir + 随机 policy** |
| exp3 | Phase 1 | ✓ | ✗ | 无 | 无 | 纯感知训练 |
| exp3 | Phase 2 | ✓ | ✓（via loss_aux） | 无 | ✓ | GRU 开始动 |
| exp4 | Phase 1 | ✓ | ✗ | 无 | 无 | 同 exp3 |
| exp4 | Phase 2 | ✓ | ✓ | 无 | ✓ | 同 exp3 |
| exp4b | Phase 2 → 3 | ✓ | ✓ → ✗（ablation） | ✗ 冻结 | ✓ | Ablation phase 撤 aux |
| exp5 | P1/P2a/P2b | ✓ | ✓ | ✓（P2b 解冻） | ✓ | 完整可训练 |
| exp6 | 全程 | ✓ | ✓ | ✓ | ✗（用 gap 代替） | 完整可训练 |

### 关键洞察

**(1) exp1 和 exp2 是 Echo State / Reservoir Computing 架构**

这是 Jaeger (2001) 和 Maass (2002) 的经典范式：**随机初始化的 recurrent 网络已经能产生足够丰富的表征，只需要训一个线性读出**。

具体到 exp2：
- `GRU`（192 隐单元，随机初始化）→ 冻结
- `W_action: Linear(192, 1)`（随机初始化）→ 冻结
- `pred: Linear(192, 4)`（唯一可训练模块）

**为什么 W_action 不用训？** 因为 W_action 生成 action 时**总是在 `with torch.no_grad():` 里**，梯度不会流回。且 pred 的 loss 只经过 `self.pred(self.h_multi)`，`h_multi` 每次都 detach。所以只有 `pred.weight` 收到梯度。

**为什么 GRU 也不用训？** 同理——`h_multi` 每次 `update_state` 结束都 `detach()`，切断梯度流。

**(2) 这个架构选择有深刻意义**

exp2 的 causal budding（ch0 spike ~13.8x）是**在最不利的条件下**证明的：
- 随机冻结的 reservoir
- 随机冻结的 policy
- 只有一个线性读出可训

**即使如此**，spike test 和 recovery 依然显示 Causal 组和 Control 组的巨大差异。**这意味着因果结构不是"学"出来的**——它是"**随机动力学 + 固定映射 + 世界反馈**"三者**自然涌现**的。pred 只是**读出**这个结构，不是**制造**它。

**科学意义**：如果 causal budding 需要精心训练的 policy，那可以质疑"是算法把因果结构塞进去的"。但**随机 policy 也够**——说明因果闭环的**发生**只需要环境+反馈+持续状态，**不需要智能决策**。这是极强的**发育性**论证。

**(3) 从 exp2 → exp6 是"逐步解冻"的架构演化**

```
exp2:      pred only trained         (最 minimal)
   ↓  加 aux head，让梯度打进 h
exp3/4 P2: pred + GRU trained
   ↓  加 policy 训练
exp5 P2b:  pred + GRU + W_action     (完整可训)
   ↓  加多策略优化
exp6:      同上 + forward sampling / direct AG / gradient disagree
```

每一步加一个可训模块，测**新增的自由度**能带来什么新能力。这个"逐步解冻"本身就是 Paper 1 方法论的一部分。

**(4) 梯度流的技术细节（`detach` 的作用）**

模型里到处出现的 `.detach()` 不是可有可无——它精确控制了**哪些权重能被训**：

```python
# exp2 SingleHeadModel
def update_state(self, obs):
    h_new = self.gru(x, self.h_gru)
    self.h_multi = ((1 - alpha) * self.h_multi + alpha * h_new).detach()  # 切断梯度
    self.h_gru = h_new.detach()                                             # 切断梯度
```

vs

```python
# exp4 ProprioceptiveModel (Phase 2 用的 step_live)
def step_live(self, obs):
    h_new = self.gru(x, self.h_gru)
    h_live = (1 - alpha) * self.h_multi + alpha * h_new    # 不 detach！保留梯度路径
    return pred, h_live
```

只在 exp3/4 Phase 2 里，`h_live` 保留梯度路径 → `loss_aux = ce(aux(h_live), ...)` 的梯度**通过 h_live 流回 GRU** → GRU 才被训练。这个 `detach` 的位置决定了架构的可训范围。

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

#### Attractor 是架构性质，不是学习产物（sanity check verified）

**核心事实**：exp1 里 GRU 权重**从头到尾没被训练**（`.detach()` 切断了梯度流，见 [训练模式对照](#训练模式对照哪些权重在动)）。整个训练循环里只有 `pred` 一个线性层在动。

**但 attractor 从哪来？** 答案是：**从架构本身来**。三层机制：

1. **GRU 的 gate + tanh 结构**保证 h 有界（凸组合 + 饱和激活）
2. **Multi-scale EMA** 强制 h_multi 落到低维流形（低通滤波，4 个 α 组）
3. **结构化输入**（正弦信号）驱动 h 跟随周期震荡

**Empirical 验证**（**完全不训练** = 一次 `backward()` 都没调）：

```
Weight delta:     GRU = 0.00e+00,  pred = 0.00e+00    ← 严格零
Test 1 Dim<30%:   5/192 = 2.6%    PASS
Test 2 Power-law: PASS
Test 3 Recovery:  97.0%           PASS  (训练版是 95.0%)
Test 5 Novelty:   peak 1.468      PASS
Test 6 Spectral:  monotonic       PASS
─────────────────────────────────────
                  5/5 PASS
```

**这 5 项 attractor 测试完全不需要训练**——**Reservoir Computing** (Jaeger 2001) 早就理论证明了：随机初始化的递归网络在合适谱半径下自然产生稳定 attractor 动力学。

**训练的真正作用**：唯一需要训练的是 **test 4（残差白噪声）**——那是测 pred 能否**充分提取**信号里的可预测结构。**训练只影响预测质量，不改变 h 的动力学结构**。

**理解 exp1 的正确框架**：
```
架构（GRU + EMA + 输入）── 免费送 ──→ 5/7 attractor 测试通过
                                        (dim, 幂律, 恢复, 新颖, 谱分离)
                                        
pred 训练              ── 换取 ──→   1/7 残差白噪声测试通过
                                        (提取信号里所有可预测结构)
```

**对 CET 框架的意义**：**"持久状态 (Persistent State)"** 作为条件 1，它的实现**不需要学习**——是纯架构条件。这是 CET §13.8 "架构通路"层面**最干净的例子**。

**核心洞察**：**训练**发生在**读出层**（pred），**动力学**发生在**架构层**（GRU + EMA）。这个分离是 Paper 1 后续所有实验能干净测量的**架构基础**。

---

### exp2: Causal Budding — 因果萌芽

**文件**：[experiments/exp2_causal.py](../experiments/exp2_causal.py)

**问题**：系统的 action 只改变 obs[0]，它能否学到"我影响的是哪个通道"？

**⚠ 架构要点**（详见 [训练模式对照](#训练模式对照哪些权重在动)）：
**exp2 是 Reservoir Computing 架构**——`GRU` 和 `W_action` 都**随机初始化后冻结**，只有 `pred: Linear(192, 4)` 被训练。causal budding 是在"最不利"架构下证明的，说明因果结构**不是学出来的，是涌现的**。

**Setup**：
- **Causal 组**：`action = f(h) = W_action · h`（**随机固定投影**，非训练所得），GAMMA=2.0
- **Control 组**：`action = AR(1) 噪声`（φ=0.95），统计匹配
- 单预测头（no dual heads yet），60K 步

**关键指标：分层证据链（三层递进，各司其职）**

exp2 的证明结构不是"多个互相验证的指标"，而是**从存在到性质到功能**的**分层论证**——每一层排除一个具体的替代解释，为下一层的可解释性奠基。

```
Level 1（存在性）：系统学到了"action 影响哪里"的模型
    ↓ ← spike test 在此层
Level 2（性质）：这个模型是因果结构，不是统计关联
    ↓ ← recovery test 在此层
Level 3（功能）：这个模型是内化的，不是外部监督假象
                    ↓ ← self-maintenance 在此层
        完整证明：causal self-world decomposition
```

**顺序不能颠倒**：Level 1 不成立，Level 2 讨论就漂浮无锚（"到底是关于什么的差异？"）。Level 2 不成立，Level 3 讨论就是空的。

---

**Layer 1 — Spike Test（通道特异性 → 存在性）**

**spike_ratio[ch] = Err_disconnected[ch] / Err_connected[ch]**

**判定**（v0.3.2 setup）：**Causal 组内** ch0 spike > 1.5x **且** > ch1, ch2, ch3

**回答的问题**："系统的预测模型里，是否有一个 built-in 的 belief 说'某个特定通道会被扰动'？"

**典型结果**（v0.3.2 setup，peak-of-first-5-steps）：
- **Causal 组**：ch0 **13.8x**，ch1 ~0.3x，ch2 ~2.5x → **组内 A ≫ B, C** ✓
- **Control 组**：ch0 **~26x**，ch1 ~11.5x，ch2 ~1x → **组内 A ≫ B, C** ✓（虽然 B 略偏高）

**Control 组也通过 spike test 是完全正确的**——因为 Control 的 action 也通过 `obs[0] += GAMMA * a` 施加到 ch0，两组都应观察到 ch0 是被扰动最强的通道。**这一层不区分 Causal vs Control，也不需要区分**——它只回答存在性。

**排除的替代解释**：
| 替代假设 | Spike 会怎样 | 实际观察 | 排除？ |
|---------|------------|---------|-------|
| A. 系统什么都没学 | 所有通道 error 随机变化 | ch0 显著、其他不动 | ✓ |
| B. 系统学到"世界很乱" | 所有通道均匀 spike | 只有 ch0 spike | ✓ |
| C. 系统学到"action 影响某处但不定位" | 多通道均匀 spike | 只有 ch0 spike | ✓ |
| D. 系统学到"action 影响 ch0" | 只有 ch0 spike | ✓ 观察到 | ✓（支持） |

**只有排除了 A/B/C**，后续讨论"这个 ch0 模型是因果的还是统计的"才有意义。

---

**Layer 2 — Long-Disconnect Recovery（因果 vs 统计 → 性质）**

**recovery = max(0, 1 − mean(E_last100) / max(E_first50))**

**回答的问题**："系统对 action 的模型是**结构化因果**的（可分解出 my_action 分量），还是**统计关联**？"

- 断开 action 后 2000 步，追踪 ch0 error 随时间变化
- 系统 h 会被"平静的 obs"持续更新，逐渐适应新分布
- **典型结果**：Causal recovery ~74.8%，Control ~57.2%
- **关键差异在这一层**：Causal 能"分解"（world = signal + my_action，拿掉 my_action 后仍能拟合 signal），Control 只有"关联"（拿不掉，只能等 h 慢慢重塑）

**排除的替代解释**：**"任何统计匹配的信号都能产生 spike"** —— Control 的 spike 存在但无法 recovery，说明它没有 Causal 那种可分解的结构化因果模型。

---

**Layer 3 — Self-Maintenance Ablation（功能 vs 假象 → 功能）**

**回答的问题**："这个自我表征是**内化的**（causal loop 上有功能价值），还是**被 aux 监督逼出来的假象**？"

- 训练完后**撤走 aux loss**，观察表征是否维持
- **典型结果**：Causal 保留 **94.9%**，Control 塌到 **53.9%**
- **决定性证据（decisive evidence）**：只有 Causal 组的表征在没有外部监督时依然被 GRU 动力学**自我维持**

**排除的替代解释**：**"self 表征只是被 aux 塞进去的"** —— 撤 aux 后如果 self 是假象，两组都应崩溃；实际只有 Control 崩溃，说明 Causal 的 self 表征已经**功能性内化**。

---

**为什么三层顺序不能颠倒**：

- 没有 Layer 1，Layer 2 的"recovery 差异"不知道是**关于什么**的差异
- 没有 Layer 2，Layer 3 的"self 表征"不知道是**因果的**还是**统计假象的**
- 三层加起来，才能完整排除所有主要替代解释，钉死"causal self-world decomposition"这个 claim

**类比**：证明"这个人是好领导者"
- Level 1：这个人是领导者吗？（有职位、做决策）
- Level 2：他的决策是有效的还是随机破坏的？
- Level 3：这是他一贯的模式还是偶然？

跳过 Level 1 就讨论 Level 3，等于**没有对象**的判断。

---

### 关键区分：权重学习 vs 状态更新（recovery 逻辑的核心）

**这个区分是理解 exp2 recovery test 因果论证的关键**——一定要分清。

|  | 权重学习（weight update） | 状态更新（state update） |
|---|-----------------------|------------------------|
| **什么在变** | `pred.weight`（Adam 优化器修改） | `h_multi`（GRU 前向 + EMA 递推） |
| **发生在哪** | 训练阶段的 `loss.backward()` + `opt.step()` | 每一步的 `model.update_state(obs)` |
| **是否需要梯度** | 是 | **否**（可以在 `torch.no_grad()` 里做） |
| **速度尺度** | 慢（60K 步梯度下降） | 快（几十步 EMA 平滑就能大幅变化） |
| **在测试阶段是否发生** | **否**（`no_grad` 冻结所有梯度） | **是**（h 持续更新） |

**exp2 完整时序 + 学习状态**：

```
================================================================
阶段 1：训练 (train_model, 60K 步)   ← 唯一有权重学习的阶段
================================================================
  ✓ pred.weight 变化（delta ≈ 1.79）
  ✗ gru.* 不变（delta = 0, grad = None）        ← reservoir
  ✗ W_action.* 不变（delta = 0, grad = None）    ← 随机固定
  * h_multi 累积到训练结束的稳态

================================================================
阶段 2：Spike test (2000 步 × 2 episodes)
================================================================
  with torch.no_grad():                          ← 全程 no_grad
    pred = model.predict()
    model.update_state(obs)
  ✗ 所有权重都不变
  * h_multi 每个 episode 从 reset_state() 归零后重新构建

================================================================
阶段 3：Long disconnect test (2700 步 = 500 warmup + 200 baseline + 2000 disconnect)
================================================================
  with torch.no_grad():                          ← 全程 no_grad
    pred = model.predict()
    model.update_state(obs)
  ✗ 所有权重都不变（包括 pred！）
  * h_multi 从"预期 action"稳态漂移到"无 action"稳态
    ← 这个漂移就是 recovery 测的对象
```

**核心事实**：**在整个 exp2 里，pred.weight 是唯一被学习的参数，且只在训练阶段 60K 步内学**。**测试阶段（spike + recovery）所有权重都冻结**，只有 `h_multi` 这个**状态**在变。

### 因果论证藏在哪里：pred.weight 的结构，通过 h 的状态漂移暴露

**如果误以为 recovery 是"权重学习"**：
- 会得出"Causal recovery 高 = Causal 训练更充分"的错误结论
- 论证退化成训练充分度差异，与因果结构无关

**正确理解（recovery 是"状态动力学"）**：
- 两组的权重都冻结，都"一样充分"训过
- **只有 h_multi 在漂移**（GRU 前向 + EMA 平滑，无梯度）
- Causal 的 h 能快速漂移到新 attractor（模型分解结构支持"设置 action 分量 = 0"这种简单变换）
- Control 的 h 漂移慢（模型是混沌统计关联，需要整体重塑）

**追问**："如果测试阶段权重不动，Causal 和 Control 的差异到底藏在哪？"

**答**：**藏在训练完后的 `pred.weight` 里**：
- Causal 的 pred.weight 学到了"如何从 causal-loop 塑造的 h 里读出 obs 预测"——对**结构化 h** 效果好
- Control 的 pred.weight 学到了"如何从 AR(1)-noise 塑造的 h 里读出 obs 预测"——对**混沌 h** 效果好
- Disconnect 后，两组 pred.weight 都不动，但 h 都需要漂移到"无 action"分布
- **Causal 因为 pred × 新 h 更接近真实 signal → recovery 高**
- **Control 因为 pred 学的是混沌关联 → 新 h 上仍残留"预期噪声" → recovery 低**

**所以：差异藏在权重的"结构化程度"里，通过状态动力学暴露出来**。这是 Paper 1 方法论最微妙的一环——**用状态动力学揭示权重里已经编码好的结构**，权重和状态在因果论证里扮演不同角色。

### Spike ~1.0x 的诊断（不是"窗口太短"）

Current `experiments/exp2_causal.py` 跑出 ch0 spike ≈ 1.0x（我们已经 empirical 验证），不是因为**窗口太短**，恰恰因为**窗口太长 + 状态重置**：

```python
for ep_name, connect_action in [("connected", True), ("disconnected", False)]:
    model.reset_state()                            # ← 每 episode 归零 h
    for step in range(SPIKE_STEPS):                # ← SPIKE_STEPS = 2000（长）
        # ...
        model.update_state(obs)
    results[ep_name] = {ch: np.mean(per_ch[ch])}   # ← 2000 步取均值
```

**两个设计淹没了 shock**：
- **`reset_state()`**：h 从 0 开始，2000 步内独立适应各自世界。**没有"稳态被突然剥夺 action"的 shock**
- **2000 步取均值**：即使有 transient shock 也被 1950 步稳态稀释

**v0.3.2 用 peak-of-first-5-steps 就能测出 spike ~13.8x**——因为它捕捉**瞬时冲击**（h 还带着"预期 action"的稳态时突然剥夺）。

**结论**：当前 code 测的是**稳态差异**（几乎为 0），v0.3.2 测的是**瞬时冲击**（显著）。**两者不是"哪个正确"，是测不同现象**。**Recovery 数字（74.8% vs 57.2%）在两个版本里都可复现**，是 Layer 2 因果性质的稳定证据。

### 两组都学到 implicit utilization，但性质不同（exp2 论证成败的关键）

**这是 exp2 最容易被忽略、也是最精妙的一点**：**Control 组不是"什么都没学"，它也学到了 implicit utilization——但性质不同**。

**训练完成时的 MSE 对比**：

```
Causal 组:   MSE ≈ 0.003    ← 学得很好
Control 组:  MSE ≈ 0.120    ← 学得不够好，但远好于随机
猜均值基线:   MSE ≈ 0.38     ← 两组都远低于这个
```

两组都**远超无脑基线**——说明**都学到了 implicit utilization**，都在利用 h 里的信息预测 obs。差别在 **utilization 的性质**：

**Causal 学到：implicit *causal* utilization**
- action = W_action(h)，是 **h 的确定性函数**
- pred(h) 可以**完美算出** action，因为都是线性映射且都从 h 出发
- 所以 pred 能**在预测里精确加上补偿项** `GAMMA · W_action(h)`
- 理论上误差 → 0（只剩信号自身噪声）

**Control 学到：implicit *statistical* utilization**
- action = AR(1) 噪声（φ=0.95），**外部生成，与 h 无关**
- 但 AR(1) 有自相关：`action_t ≈ 0.95 · action_{t-1} + 新噪声`
- h 通过 obs 反馈**间接**知道 action_{t-1}（obs_{t-1} 里有它的效果）
- 所以 pred(h) **可以**预测 action_t 的 95%（利用自相关）
- 但每步 5% 的**新噪声**是**根本不可预测**的 → 误差有下限（MSE ~0.12）

### 为什么 Control 学到"错的东西"反而让 exp2 论证有力

**如果 Control 什么都没学**：
- 对比就 trivial：Causal 学到 vs Control 没学到
- 只能证明"训练确实有效"
- **无法证明 Causal 学到的是"因果"**（无对照可比）

**Control 学到 statistical utilization 让论证成立**：
- 两组**都通过 implicit utilization 大幅降低了 error**（都过关）
- **通道特异性**（spike test）两组都通过 ✓
- **但性质不同**：Causal 可分解，Control 纠缠
- **Recovery test 用干预暴露这个性质差异**

### Recovery 差异的机制（现在完整了）

**Causal（可分解模型）**：
```
pred 学到的映射：obs[0] ≈ signal_component(h) + action_component(h)
                                                 ↑ 从 h 提取 W_action(h) · GAMMA

Disconnect 后 world 只有 signal：
  h 漂移到"无 action 效果"分布
  在新 h 上 pred：signal_component(new_h) + action_component(new_h)
                                             ↑ new_h 里 action pattern 微弱，自动 → 0
  → 恢复 74.8%
```

**Control（纠缠模型）**：
```
pred 学到的映射：obs[0] ≈ predicted_signal(h) + predicted_noise_continuation(h)
                              ↑ 混着 signal 和 AR(1) 一起学

Disconnect 后 world 只有 signal（AR(1) noise 突然消失）：
  h 需要漂移到"noise = 0"的分布
  但 pred 学的是"noise 延续"—— 在新 h 上仍在预测"应该有的 noise 延续"
  世界不给 noise → 系统性误差
  → 恢复 57.2%（较差）
```

**关键差别**：
- Causal 的 pred 学到"**两项可分离**"→ disconnect 只需让其中一项自动清零 → **只需 h 漂移就够**
- Control 的 pred 学到"**noise 延续性**"→ disconnect 打破延续性 → **h 漂移不能修复 pred 的错误预期**

### 通俗类比

**Causal**：像**手风琴演奏者**——学到"我按左手键→声音变化"，如果左手被禁用，他知道去掉左手声部就行

**Control**：像**收音机听众**——学到"这个电台通常吵闹"，如果电台突然静音，他还在预期背景噪声，需要更长时间调整期待

**两个都学会了"声音的利用"**（否则连"什么是安静""什么是吵闹"都分不清）。但一个学到的是**因果结构**（可分解、可干预），一个学到的是**统计规律**（延续性、期望值）。

**Recovery test 通过"突然拿掉声音"暴露区别**——演奏者秒适应，听众还在等吵闹回来。

### exp2 方法论最精妙的一环

**不是"Causal 学到 vs Control 没学到"**（太简单）

**而是"两种学习性质不同——一个可分解因果模型，一个纠缠统计关联"**

**Control 学到"错的东西"，反衬 Causal 学到"对的东西"**——如果 Control 什么都没学，反衬就没有力度了。**Control 的"部分成功"是论证的必要条件**。

---

**关键代码**：
- 训练：[exp2_causal.py:87](../experiments/exp2_causal.py#L87) `train_model()`
- Spike 测试：[exp2_causal.py:143](../experiments/exp2_causal.py#L143) `spike_test()`
- 长断连：[exp2_causal.py:202](../experiments/exp2_causal.py#L202) `long_disconnect_test()`
- Self-maintenance：[exp4b_self_maintenance.py](../experiments/exp4b_self_maintenance.py)

**在证据链中的位置**：**步骤 2 + 3 + 6**（Pearl Layer 2 干预 + 混淆控制 + self-maintenance 功能验证——三层加起来是 exp2 完整证据）

---

### ⚠ exp2 的严格边界：Implicit Causal Utilization，不是 Self-World Decomposition

**这是 exp2 最大的一处推理跳跃，必须自己钉清楚。** Recovery test 严格能挣到的结论，比"self-world decomposition"弱一层。

**先分清哪些证据是"两组共享"的**（别把它们误当 Causal 判据）：ch0 的**通道特异 spike**（Causal ~13.8x、Control ~26x）和 **ch0 预测质量**，**两组都有**——因为 Control 的 action 也经 `obs[0] += GAMMA·a` 施加到 ch0。它们只证明"系统学到 action 影响 ch0"这个**存在性现象**，**区分不了因果 vs 统计**。**两组都做 implicit utilization**；exp2 里**唯一**把 Causal 从 Control 分开、唯一给"causal"这个定语背书的判据，是 **long-disconnect recovery**（Causal = 可复现的因果利用；Control = 纠缠的统计利用）。

**Recovery 排除了什么、留下了什么**：
- Control（匹配统计）recovery 更差 → **排除了"纯统计相关"**（记为 C：任何统计匹配的信号都行）
- 但剩下 **A（表征性分解：系统内部区分"我造成的" vs "世界造成的"）vs B（只是保留了一个自相关、有预测价值的内部变量）** —— **Recovery 对 A/B 无能为力**，两者都符合观测。

**"decomposition" 一词的两处偷换**（overclaim 的来源）：
1. exp2 真正分解的轴是 **"可从 h 预测 / 不可预测"**，action 落在可预测侧**只因为实验者把它接成 `action = W_action(h)`**。把"可预测分量"贴上 **"self"** 标签的是**实验者**，不是系统——系统没有画出"我/世界"的边界。
2. pred.weight 里的**机制性分解**（预测拆成 signal 分量 + action 分量）≠ h 里的**表征性/可读分解**（能读出"我在动"）。exp2 至多有前者。

**代码佐证**：`long_disconnect_test` 断开阶段（[exp2_causal.py:250](../experiments/exp2_causal.py#L250)）**根本不调用 `get_action()`**——recovery 差异是**训练时 pred.weight 学到的结构 + h 状态漂移**的属性，连"内部动作持续生成"都算不上，更支持 utilization 读法。

**Paper 自己的承认**：**exp3 的 trailing recall ≈ 12%** 就是量化证据——exp2 那种因果利用**读不出**"我在动"。所以"如果 exp2 已证明 decomposition，exp3/4/5 就多余了"——它们的存在本身反证了 exp2 未达 self-representation。

**严格表述**（实验与结论完全对齐）：
> The predictive system retains an **internally regenerable** dependence on self-generated action, beyond what matched statistics alone can explain.

——只有 *internal / regenerable dependence*，没有 *self representation*，没有 *causal decomposition*。"self-world decomposition / self-representation" 应留到 **exp4（可读 trace）+ exp4b（自维持）** 才动用。参见 [八步证据链](#八步证据链完整版) 第 4 步与 [encoding gap](#exp3-encoding-gap--编码鸿沟)。

---

### exp3: Encoding Gap — 编码鸿沟

**文件**：[experiments/exp3_encoding_gap.py](../experiments/exp3_encoding_gap.py)

**问题**：即使 aux head 强推 h 编码"我在动作"，只有 obs(4d) 输入的 GRU 能不能做到？

**⚠ 训练模式**（详见 [训练模式对照](#训练模式对照哪些权重在动)）：
- **Phase 1（100K 步）**：与 exp2 同样是 reservoir 架构——**只训 `pred`**
- **Phase 2（80K 步）**：加入 aux head，梯度通过 `h_live` **流回 GRU**——**pred + GRU + aux 都被训练**
- 关键差别在 `step_frozen` (Phase 1，`h_multi.detach()`) vs `step_live` (Phase 2，`h_live` 不 detach)

**Setup**（关键：**没有 trace**）：
- GRU 输入 = obs(4d) only
- Aux head 三分类（active / trailing / quiet），梯度打进 h
- **BinaryProbe 用 detached h** 测 trailing 时段能否读出

**BurstGate 机制**：随机开关 action bursts，产生 active/trailing/quiet 三个 phase。`trailing` 是 action 刚停后 50 步——测的是"h 能不能在 action 消失后还记得刚才动过"。

**期望结果**：**trailing recall ≈ 12%**（接近随机，encoding gap 存在）

**关键代码**：[exp3_encoding_gap.py:73](../experiments/exp3_encoding_gap.py#L73) `Model` (4d GRU), [exp3_encoding_gap.py:115](../experiments/exp3_encoding_gap.py#L115) `BinaryProbe`

**核心洞察**：pred_A 能预测好（隐式补偿），但 h 里没有"我在动"的可读维度。**这就把"prediction quality 好"和"self-representation 存在"在同一个模型里分离开了**。

**在证据链中的位置**：**步骤 4**（编码分离，与 exp4 对照）

### ⚠ exp3 的诚实定位：不是"惊天发现"，是"对照设置"

**"能用 ≠ 有表征" 这个观察，对熟悉神经网络的人是常识**（distributed representation, Hinton 1986）——线性 probe 读不出分布式特征本来就是老生常谈。

**exp3 单独看没有强 novelty**——如果只是"我们发现了 encoding gap"，ML 审稿人会说"当然，这就是分布式表征"。

**exp3 的真正角色是"baseline for contrast"**——**为 exp4 建立对照**。价值在于：
- 先严格测出**"在这个具体最小 setup 下 gap 存在"**（recall 12%）
- 再让 exp4 展示**"最小改动就能跨过"**（recall 60%+，只加一维 trace）
- 两者的**差异幅度**（12% → 60%+）+ **最小性**（1 维 EMA 就够）才是原创贡献

**独立于对照，encoding gap 本身不是 Paper 1 的核心发现**。**真正的核心发现在 exp4**：**跨越 gap 的最小充分条件是什么**。

### 三种读者的不同视角（audience 差异）

**ML 视角**：encoding gap 是分布式表征的常识 → exp3 走过场 → 关键在 exp4 找到最小 fix

**认知科学 / 哲学视角**：encoding gap 是新的、令人惊讶的（"预测能力和自我表征分离"）→ exp3 有独立价值

**Paper 1 隐含地对第二类受众说话**——这是它跨学科的代价。**审稿人质疑"encoding gap 有什么新的"时，正确回应不是"很新"，是"gap 本身不新，最小充分构造是新的"**。

详见 [误读 15：encoding gap 是 Paper 1 的核心新发现](#误读-15encoding-gap-本身是-paper-1-的核心新发现)。

---

### exp4: Proprioceptive Breakthrough — 本体感受突破

**文件**：[experiments/exp4_proprioception.py](../experiments/exp4_proprioception.py)

**问题**：加一维 trace（|action| 的 EMA）能否打破 encoding gap？

**⚠ 训练模式**：**与 exp3 完全相同**——Phase 1 只训 pred，Phase 2 训 pred+GRU+aux。**唯一差别是 GRU 输入维度 4d → 5d**。

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

## 证据三元组速查表

**这一章是紧凑参考**，把 §7 分实验详解里散布的信息压缩成三元组：**claim → 证据 → 排除的替代解释 + 未能证明什么**。特别是最后一列（未能证明）是保持严格边界的关键——**Paper 1 每个实验都有明确的证明力上限**。

### 总表（at-a-glance）

| # | 实验 | 核心 claim | 主要证据 | 关键排除 | 未证明 |
|---|------|-----------|---------|---------|--------|
| 1 | **exp1** | h 能形成稳定 attractor（架构性质） | 7 项 scorecard 6/7 pass；5 项 attractor 测试**未训也全过** | h 是白噪声 / 会发散 / 单一时间尺度 / 无响应 | attractor **对下游任务有用**（需训练验证）；**承载 self 内容**（这是 exp2+ 的事） |
| 2 | **exp2** | Implicit Causal Utilization 涌现 | 三层证据：spike 通道特异 + recovery 74.8% vs 57.2% + self-maint 94.9% vs 53.9% | 系统没学 / 影响所有通道 / 任何统计匹配信号都行 / self 是 aux 塞的 | **Self-World Decomposition**（只到 Utilization）；A vs B ambiguity 在 h 层开放 |
| 3 | **exp3** | 能用 ≠ 能读出（Encoding Gap） | trailing recall **12%**，即使 aux 使劲推 | 训练不够 / probe 太弱 / action shortcut | "更长训练 / 非线性 probe 或许能读到"（未测） |
| 4 | **exp4** | 加一维 trace 就跨过 gap | 与 exp3 同 setup，只差一维 → **12% → 60%+** | trace 是 label shortcut / 训练时长差异 / aux 效果更强 | **"1D trace 是最小充分"是 empirical，非理论最小**；**任意 vs 特定信号 ablation 未做** |
| 5 | **exp4b** | 表征只在因果有用时自维持 | 撤 aux 后 Causal **94.9%** vs Control **53.9%** | 训练过的表征都自维持 / 时长差异 / aux 是唯一支撑 | 表征"**有意识**"（纯选择效应）；**长期稳定性**（只测 5000 步）；signal-agnostic（未测） |
| 6 | **exp5** | 训练时序是 identifying assumption | 三种同步 LR 都失败；async 双通过 | LR 问题 / 架构问题 / 训练不足 | **async 是唯一可行时序**（未测其他序列）；**必须严格 3-phase**（未测其他分段） |
| 7 | **exp6** | 定量 agency + Pearl Layer 3 反事实 | AG > 0 + counterfactual spike **17.32×** + Lorenz 鲁棒 + delay 鲁棒 | pred_A 只把 action 当装饰 / 只能瞬时反馈 / 只在正弦下 | **AG 是 predictive advantage，非严格 causal advantage**（Paper 1 §7 承认此 limitation） |

### 详细三元组

**exp1 — Perception**
- **Claim**：h 能形成低维稳定 attractor（**架构性质**，不依赖学习）
- **Evidence**：Dim < 30% (2.6%) / Recovery 97% / Novelty 1.47 / Spectral separation / Power-law → **完全不训练也全过**；第 4 项残差白噪声需 pred 训练
- **排除**：h 是白噪声 / 会发散 / 只有单一时间尺度 / 对输入无响应
- **未证明**：attractor 对下游任务有用；承载 self 内容

**exp2 — Causal Budding**
- **Claim（严格版）**：Implicit Causal Utilization——不是 Self-World Decomposition
- **Evidence（三层）**：
  - **Layer 1 存在性**：spike test（两组都通过 = 通道特异性）
  - **Layer 2 性质**：recovery test（Causal 74.8% > Control 57.2% = 结构化 vs 统计）
  - **Layer 3 功能**：self-maintenance（Causal 94.9% vs Control 53.9% = 内化 vs 假象）
- **排除**：什么都没学 / 影响所有通道 / 任何统计匹配信号都行 / self 是外部塞的
- **未证明（严格边界）**：
  - **Self-World Decomposition**（只到 Utilization——用了因果，不代表分解了 self/world）
  - **A（表征分解）vs B（保留 self-correlated 变量）**在 h 层的区分（recovery 只测 obs 层）
  - Spike 不区分 Causal vs Control（两组都通过——只测通道特异性）

**exp3 — Encoding Gap**
- **Claim**：能用 ≠ 有可读表征（Paper 1 核心概念发现）
- **Evidence**：即使 aux head 主动施压 h 编码"我在动"（三分类 + 梯度回流），probe 读 detached h 依然只有 **12%**（≈ chance）
- **排除**：
  - "训练不够" ← aux 使劲推 80K 步还是 12%
  - "probe 太弱" ← pred 也是线性 Linear(192,4)，能预测好；**同类型 readout 里 pred 成功 probe 失败** = **不是 readout 能力问题**
  - "action shortcut" ← **exp3 里 action 是纯随机 uniform(-2, 2)**，不是 W_action(h)，h 不能天然含 action
- **未证明**：更复杂 probe / 更长训练能否读出（Paper 1 只测线性 probe + 80K 步）

**exp4 — Proprioceptive Breakthrough**
- **Claim**：加一维 trace 就能跨过 encoding gap
- **Evidence**：与 exp3 **完全同 setup**（random action、同 aux、同 probe），唯一差别 GRU 输入维度 4d → 5d（多一维 τ = EMA(|a|)）→ trailing recall **12% → 60%+**
- **排除**：
  - "trace 是 label shortcut" ← trace 是 |action| 的 EMA，不携带 is_trailing 标签
  - "训练时长差异" ← 完全相同的训练步数
  - "aux head 效果更强" ← aux head 完全相同
- **未证明（Paper 1 里可补的洞）**：
  - **"1D trace 是最小充分"是 empirical 观察**，理论最小可能更小
  - **"trace-specific vs 任意信号"没做严格 ablation**（应该测"加一维随机噪声"或"加一维不相关信号"作为对照）——严格版验证 trace 的**信息内容**才是必要的，而不只是"多一维输入"

**exp4b — Self-Maintenance**
- **Claim**：h 里的 self 表征**只在因果有用时**被 GRU 动力学自我维持
- **Evidence**：训完 Phase 2（Causal 95.3%, Control 91.9%——两组都被 aux 训到高精度），撤 aux + 冻结参数 5000 步：Causal **94.9%**（几乎不变），Control **53.9%**（接近 chance）
- **排除**：
  - "训练过的表征都自维持" ← Control 塌到 53.9% 排除
  - "训练时长/参数量差异" ← 两组完全相同
  - "aux 是唯一支撑" ← 两组撤 aux 都失去支撑，只有 Control 塌
- **未证明**：
  - 表征"**有意识**"（纯粹是选择效应）
  - **长期稳定性**（只测 5000 步）
  - **signal-agnostic**（只在特定 3 通道 setup 下测过）

**exp5 — Asynchronous Awakening**
- **Claim**：训练时序（perception 稳定后再解冻 action）是 **identifying assumption**（必要条件，不是超参）
- **Evidence**：
  - **Async**（3-phase 分阶段）：spike 5.58×, trailing 66.3% ✓ 双通过
  - **Sync FAST**（LR 1e-3）：4.76×, 60.5%（通过但不如）
  - **Sync MEDIUM**（LR 5e-4）：3.98×, **21.5%**（trailing 失败）
  - **Sync SLOW**（LR 1e-4）：2.52×, 40.3%（marginal）
- **排除**：LR 问题 / 架构问题 / 训练不足——三种 LR 都不如 async，架构完全相同
- **未证明**：async 是唯一可行时序 / 必须严格 3-phase / 其他信号或架构下也成立

**exp6 — Agency Gain Measurement**
- **Claim**：定量测量 agency + 用 counterfactual 达到 Pearl Layer 3
- **Evidence**：
  - AG = Err_B − Err_A > 0（sinusoidal pred gap 80.7%, Lorenz 99.5%）
  - **Counterfactual spike**（喂 pred_A 假 action）：spike_zero = 17.32×
  - **Phase 1 vs Phase 2b spike 分离**：0.95× vs 17.32× —— pred gap 大不等于 causal 依赖（Phase 1 gap 98.8% 但 spike ≈ 1，是机械补偿）
  - 跨信号（Lorenz）+ 延迟（2 步）鲁棒
- **排除**：pred_A 只把 action 当装饰 / 只能瞬时反馈 / 只在正弦下
- **未证明**：AG 是 predictive advantage，**不是严格 causal advantage**（Paper 1 §7 limitation 明确承认——严格 causal 需要完整 Pearl do 算子框架）

### Claim 强度分级

Paper 1 每个实验的证明力有明确上限，分级如下：

| 实验 | Claim 强度 | 严格边界 |
|-----|:--------:|---------|
| exp1 | 中 | attractor **存在**，但不涉及 self |
| exp2 | 中偏弱 | Implicit Causal Utilization，**不是 SW-Decomposition** |
| exp3 | **强**（negative claim） | 能用 ≠ 能读，可靠地**证否**了"能用即能读" |
| exp4 | 强 | 最小充分构造，但**缺"1D 是否最小"的 ablation** |
| exp4b | 强 | 选择效应清晰，但**只测短期**（5000 步） |
| exp5 | 强 | 时序必要性明确，但**未测其他可能时序** |
| exp6 | 强 | Pearl Layer 3 + 定量 + 跨信号 + 延迟鲁棒 |

**规律**：**negative claims（如 exp3 的"读不出"）比 positive claims 更强**——因为证否只需要一个反例的**不存在**，而证实需要**排除所有替代解释**。

### 关键洞察：Self-World Decomposition 需要五个实验合力

**Paper 1 没有任何单一实验能独立证明完整的 Self-World Decomposition**。完整证明需要**五个实验合取**：

1. **exp2 pred gap** → $D_{KL}^{obs}$ 吸收（相关侧面）
2. **exp2 recovery + spike** → $D_{KL}^{int}$ 吸收（因果侧面，obs 层）
3. **exp4 probe recall** → h 层可读表征（**把 A 从 B 中分离**——这是关键的第 3 条）
4. **exp4b self-maintenance** → 表征功能内化（全局选择效应）
5. **exp5 async** → 发育时序满足（identifying assumption）

**任何一个缺失，Self-World Decomposition 都不成立**。这正是 Paper 1 采用"发育序列"叙事的原因——**结论是合取的，实验必须联合起来读**。

---

**这份速查表的两个用法**：

1. **快速回忆**：读 §7 详解后忘了某个实验证明什么，来 §8 看一行
2. **保持严格**：写论文 / 回复审稿人时，看"未能证明"一列——**不要越过 Paper 1 实验的严格边界 oversell**

**审稿人问"你的实验证明了 X 吗"，看"claim"列决定是不是；看"未证明"列决定要不要收缩**。

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

**AG > 0 是构造上必然的**（pred_A 拿 action、pred_B 不拿）。AG 的作用是**量化标尺**，不是因果证据。真正证明 agency 需要**分层证据链**：
- **Layer 1 存在性**：spike test → 通道特异性（两组都通过是 desired）
- **Layer 2 性质**：recovery → 因果 vs 统计（只有 Causal 通过）
- **Layer 3 功能**：self-maintenance → 内化 vs 假象（decisive evidence）
- **Layer 4 显式表征**：encoding gap → 因果利用 vs 自我表征

**AG 是横跨所有层的量化标尺，不替代任何一层的证据**。详见 [exp2 章节的分层证据链](#exp2-causal-budding--因果萌芽) 和 [误读 11](#误读-11spike-test-应该唯一区分-causal-vs-control)。

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

### 误读 8："exp2 里 W_action 是训练出来的最优 policy"

**不是**。exp2 里 `W_action` 是**随机初始化后永远不动**——它的输入在 `torch.no_grad()` 里，梯度不流回。整个 exp2 只有 `pred: Linear(192, 4)` 一个模块在学。这是 **Reservoir Computing** 架构：随机冻结的 GRU 提供高维特征，随机冻结的 W_action 提供固定策略，唯一的可训权重是线性读出。

**这个事实反而加强了 exp2 的结论**：causal budding 在最不利的架构条件下依然涌现——因果结构**不是学出来的，是随机动力学 + 世界反馈自然产生的**。详见 [训练模式对照](#训练模式对照哪些权重在动)。

### 误读 9："测试期间系统还在学习"

**不是**。所有 spike test、long-disconnect test、probe recall 测试**全程在 `torch.no_grad()` 里**——权重完全冻结。**h_multi 状态缓冲区**会更新（这是状态动力学，不是权重学习）。

Recovery 测的是**状态适应**——系统的 h 能否被"新分布的 obs"重塑成新的 attractor，而不是权重能否重新学习。

### 误读 10："exp1 的 6/7 scorecard 是通过训练学出来的"

**错**。这是**最反直觉的一条**，但已经被 empirical 严格验证：

**完全不训练**（GRU 权重 delta=0, pred 权重 delta=0）跑 exp1 setup，5 项 attractor 测试**全过**：
- Dim < 30%: **2.6%** PASS
- Power-law: PASS
- Recovery: **97.0%** PASS（比训练版 95.0% 还略高）
- Novelty: 1.468 PASS
- Spectral separation: PASS

**为什么？** 因为这 5 项测的都是**架构性质**：
- 低维、稳定、多尺度、有响应——都是 **GRU 有界动力学 + Multi-scale EMA 低通滤波 + 结构化输入**三者的**自然产物**
- 这是 Reservoir Computing (Jaeger 2001) 的经典结论

**训练的真正作用**：只影响 **test 4（残差白噪声）**——测 pred 层能否**充分提取**信号里所有可预测结构。这是 pred 的学习成果，不改变 attractor 结构。

**理解 exp1 的正确框架**：
- **架构**（GRU + EMA + 输入结构）→ 免费送 5/7 attractor 测试
- **pred 训练** → 换取 1/7 残差白噪声测试

**这个区分对 CET 框架非常重要**：条件 1（Persistent State）的实现**不需要学习**，是纯架构条件。详见 [exp1 章节的 sanity check 小节](#exp1-perception--稳定-attractor)。

### 误读 11："Spike test 应该唯一区分 Causal vs Control"

**错。这是最容易导致错误质疑 exp2 结论的一条误读。**

**Spike test 不是干这个的**。它测的是**通道特异性**（Level 1：存在性）——回答"系统的预测模型里，是否有一个 built-in 的 belief 说'某个通道会被特定扰动'？"

**关键事实**：**两组都应该通过 spike test**——因为两组的 action 都通过 `obs[0] += GAMMA * a` 施加到 ch0，两组的系统都应观察到 ch0 是被扰动最强的通道，都应形成"ch0 特殊"的预期。

- **Causal ch0 spike 13.8x** ✓（组内 A ≫ B, C）
- **Control ch0 spike 26x** ✓（组内 A ≫ B, C）

**Control 的 spike 甚至更高**——这不是"spike 无效"的证据，只是说明 AR(1) 外部噪声比 W_action(h) 内部结构化行为更**不可预测**，所以断开时冲击更大。**这依然通过通道特异性判定**。

**如果误以为 spike 应该"只 Causal 有、Control 没有"**，就会得出"exp2 不成立"的错误结论。**这是把证据链 Level 1（存在性）和 Level 2（因果性质）的角色搞混了**。

**正确的分层理解**（三层都是 Pearl Layer 2 干预，但建立不同 claim）：
- **Spike test**（证据链 Level 1）：证明两组都学到了"action 影响 ch0"这个通道特异性模型（**存在性**）
- **Recovery test**（证据链 Level 2）：证明只有 Causal 组的模型是结构化因果的，Control 只是统计关联（**性质**）
- **Self-maintenance**（证据链 Level 3）：证明 Causal 的表征是内化的，不是 aux 的假象（**功能**）

**必须先建立 Level 1，Level 2 的差异才有可解读性**——否则"两组 recovery 不同"这个事实不知道是关于什么的差异。详见 [exp2 章节的分层证据链](#exp2-causal-budding--因果萌芽)。

### 误读 12："Recovery 是权重学习的过程"

**错**。recovery test 全程在 `torch.no_grad()` 里——**所有权重都冻结，包括 pred**。

在 disconnect 阶段：
- pred.weight 不动
- gru.* 不动（本来就没训过）
- W_action.* 不动（本来就没训过）
- **只有 `h_multi` 在通过 GRU 前向 + EMA 递推持续更新**

**Recovery 测的是"状态动力学"**——h_multi 能否被"新分布的 obs"重塑成新的 attractor，而不是权重能否重新学习。

**这个区分决定了 recovery 的因果论证方向**：
- 权重都冻结、都一样"充分"训过
- Causal 和 Control 的**差异藏在训练结束时的 `pred.weight` 结构里**
- Disconnect 后**通过 h 的状态漂移暴露出来**
- **能不能快速漂移到低 error 的新 attractor** = 权重结构是"结构化因果"还是"混沌关联"

详见 [exp2 章节的"权重学习 vs 状态更新"](#exp2-causal-budding--因果萌芽)。

### 误读 13："Control 组什么都没学"

**错**。Control 组也学到了 implicit utilization——只是**性质不同**。

- Causal MSE ≈ 0.003（学得很好）
- Control MSE ≈ 0.120（学得不够好但远超无脑基线 ~0.38）

两组**都远超无脑基线**——都学到了利用 h 里信息预测 obs。**差别在 utilization 的性质**：

| | Causal | Control |
|---|-------|--------|
| **学到的 utilization 类型** | *causal*（因果）| *statistical*（统计）|
| **依据** | action = W_action(h)，从 h 完全可算 | action = AR(1) 噪声，从 h 只能预测 95%（自相关） |
| **模型结构** | 可分解（signal + my_action） | 纠缠（signal 和 noise 混在一起）|
| **训练 MSE** | ~0.003（近乎完美） | ~0.12（有下限，5% 新噪声不可预测）|

**为什么这个 subtle 但关键**：如果 Control 什么都没学，exp2 的对比就 trivial 了——只能证明"训练有效"。**Control 学到"错的东西"（statistical utilization）反衬 Causal 学到"对的东西"（causal utilization）**——这才是 Recovery test 能干净区分因果 vs 统计的关键。

**Control 的"部分成功"是 exp2 论证的必要条件，不是缺陷**。详见 [exp2 章节的"两种 utilization 性质区分"](#exp2-causal-budding--因果萌芽)。

### 误读 14："Pearl 分层 = 证据链分层"

**错**。这是两个正交的分层轴，一定要分清：

- **Pearl 三层**：**查询类型**（用了什么因果操作）——观察 / 干预 / 反事实
- **证据链三层**（本手册）：**claim 类型**（建立了什么结论）——存在性 / 性质 / 功能

**Spike、Recovery、Self-maintenance 在 Pearl 意义上都是 Layer 2 干预**（都做了 do 操作）。它们在**证据链上**扮演不同角色（存在 / 性质 / 功能），但**不是通过升级 Pearl 层次**来实现的，而是通过**不同的干预设计 + 不同的比较策略**。

只有 **exp6 的 counterfactual spike**（"喂 pred_A 假 action"）是真正的 Pearl Layer 3。

**为什么这个区分重要**：如果把两者混淆，会误以为"证据链升级 = Pearl 层次升级"，从而得出错误结论（如"self-maintenance 是 Layer 3 因为它在证据链最高"——不对，它也是 Layer 2 干预）。

详见 [方法论核心章节](#方法论核心干预式因果推断)。

### 误读 15："Encoding gap 本身是 Paper 1 的核心新发现"

**部分错**。**"能用 ≠ 有表征"从 ML 视角看是常识**（distributed representation 里线性 probe 读不出 是老话题，Hinton 1986 起）。**Paper 1 的核心 novelty 不在这里**。

**真正的原创贡献分层**：

| 层次 | 内容 | Novelty |
|-----|------|:-------:|
| ❌ Encoding gap 本身存在 | 分布式表征让线性 probe 失败 | 低（40 年常识） |
| ⚠ 具体化到 self-representation | 把 gap 应用到"我在动"这个自我概念 | 中（跨学科桥接） |
| ✓ **最小充分条件（跨越 gap）** | **1D trace（EMA of \|a\|）就够** | **高（原创）** |
| ✓ **构造性 + 量化 + 发育框架** | **4 conditions, 12 falsified alternatives, developmental order** | **高（方法论）** |

**Paper 1 Section 6.2 的表述**"encoding gap is perhaps the most important finding for theories of self-awareness" **在 ML 视角看有点 oversell**。**更严格的表述**：

> "The dissociation itself is expected from distributed representation theory. Our contribution is the **minimal constructive characterization**: identifying the exact architectural condition (1D proprioceptive channel of specific form) that is jointly sufficient for this dissociation to close in a minimal artificial system."

**审稿人问"encoding gap 有什么新的"，正确回答**：**"gap 本身不新，最小充分条件是新的"**。

**exp3 的正确定位**：不是"发现 gap"，是**"为 exp4 建立对照 baseline"**——**12% → 60%+ 的对比 + trace 的最小性**才是原创。

详见 [exp3 的诚实定位](#exp3-encoding-gap--编码鸿沟)。

### 三种受众的不同视角（audience mismatch）

Paper 1 是**跨学科工作**，不同背景读者对 encoding gap 的反应差异很大：

| 受众 | 对 encoding gap 的反应 | 应对策略 |
|-----|---------------------|---------|
| **ML 专家** | "这是分布式表征常识，有什么新的？" | 强调**最小充分条件**（1D trace）+ 发育框架 |
| **认知科学 / 哲学** | "predictive competence 和 self-representation 分离！新颖" | 强调**具体化到 self**这个跨学科桥接 |
| **神经科学** | "对应 deafferentation 现象，可控 minimal 版本" | 强调**生物学对应**（trace ≈ proprioception） |

**Paper 1 的叙事默认对第二、三类受众设计**——**对 ML 受众显得 obvious，这是它跨学科的代价**。**当被 ML 审稿人质疑时，把叙事重心转到"最小构造 + 框架"上**。

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

## 用 CET 重读 Paper 1

Paper 1 是 CET (Constraint Emergence Theory) 框架下 **§13.8 "Self 涌现的信息论解释：条件集扩张"** 的最小 empirical demonstration。本章用 CET 的精确术语重读 Paper 1 的核心实验，并修正前面章节中一些用词不严格的地方。

**前置说明**：CET 有配套论文 ORI (Observer-Relative Information)，$I_{ORI} = I_{CET} = \mathbb{E}_{S_t}[D_{KL}(P_{true}(S_{t+1} \mid S_t) \| P_M(S_{t+1} \mid S_t))]$——是同一个量，两种解读侧重（"观察者相对信息" vs "约束与模型的落差"）。本章以 CET 术语为主。

### 12.1 §13.8 视角：Paper 1 的发育序列 = 条件集扩张

**CET §13.8 的核心命题**：Self 不是先验假设，而是**系统在持续降低 $I_{CET}$ 过程中，逐步将预测条件集扩张到包含自身状态**的结构性解。每一次扩张需要**两个条件同时满足**：

1. **理论必要性**：$I(X;\, S_{t+1} \mid \text{已有条件集}) > 0$——变量 X 携带独立预测信息
2. **架构必要性**：系统拥有获取 X 的**信息通路**

**Paper 1 的四个 sufficient conditions 精确对应 Level 0→3 跨越所需的架构通路**：

| Level 跨越 | 纳入变量 | 理论条件 | 架构通路（Paper 1 实现） | 验证实验 |
|-----------|--------|---------|----------------------|---------|
| 0 → 1 | $S_t$（世界状态）| $I(S_t; S_{t+1}) > 0$ | GRU + multi-scale EMA（**persistent state**）| exp1 scorecard |
| 1 → 2 | $A_t$（行动）| $I(A_t; S_{t+1} \mid S_t) > 0$ | obs[0] += GAMMA·a（**causal action loop**）| exp2 spike + recovery |
| 2 → 3 | $M_t$（内部状态）| $I(M_t; S_{t+1} \mid S_t, A_t) > 0$ | trace = EMA(\|a\|)（**proprioceptive feedback**）| exp3 vs exp4 recall gap |
| — | 序 | 顺序不可交换 | 分阶段解冻（**async awakening**）| exp5 sync vs async |

**Encoding gap 的 CET 精确定义**：

**Level 2 → 3 跨越的第二条件（架构通路）缺失时的必然现象**——系统有 $I(M_t; S_{t+1} \mid S_t, A_t) > 0$ 的理论条件（$M_t$ 携带独立预测信息），但没有本体感受通路让 $M_t$ 进入预测计算。**优化压力存在但无法作用**，Level 停在 2。

- **exp3**：GRU 输入 4d，缺少 $M_t$ 通路 → probe recall 12%（Level 卡在 2）
- **exp4**：加一维 trace（打开通路）→ probe recall 60%+（Level 3 跨越）

**这个理解重塑了 encoding gap 的意义**：它不是"新发现"（ML 常识层面 distributed representation），而是 **§13.8 第二条件缺失的必然预测**——Paper 1 的价值在于证明**这个理论预测在最小人工系统里可复现，且给出跨越的最小充分通路**。

### 12.2 §9.8 视角：Paper 1 的关键指标 = 约束的两把尺

**CET §9.8 的核心命题**：**约束 $\mathcal{C}$ 只有一套（客观），但可以从两个侧面被探测**：

- **$D_{KL}^{obs}$（观测偏差）**：被动接收数据，探测约束的**相关侧面**——需要 Axiom 1+2+3
- **$D_{KL}^{int}$（因果偏差）**：主动执行 $do(A)$，探测约束的**因果侧面**——需要 Axiom 1+2+3+**4**

**没有 Axiom 4（行动），因果侧面从认识论上被永久封死**——这是"为什么因果推理需要 do 算子而不只是相关分析"的严格解释。

**Paper 1 的关键指标对应两把尺**：

| 指标 | CET 探测类型 | 探测约束的哪个侧面 |
|------|:----------:|-----------------|
| **pred gap**（93.7%）| $D_{KL}^{obs}$ | 相关侧面（知道 action 后预测改善多少）|
| **spike test**（v0.3.2: 13.8x）| $D_{KL}^{int}$ | **因果侧面**（$do$(断开 action) 后系统响应）|
| **recovery test**（74.8%）| 持续 $D_{KL}^{int}$ | 因果侧面（长期干预下的结构性适应）|
| **Control 组对比** | 两种探测的交叉验证 | 区分相关 vs 因果 |
| **exp6 counterfactual** | h 层 $D_{KL}^{int}$ | 因果侧面（在 pred_A 输入端做反事实）|

**为什么 spike test 不能用相关性替代**：CET §6.3——相关性是对称的（$I(X; Y) = I(Y; X)$，无法区分方向）。**do 算子是切断所有指向被干预变量的箭头**，只有干预实验能触及约束的因果侧面。

### 12.3 §6.4-6.5 视角：Pearl 局部因果 vs CET 全局因果

**CET §6.4 的核心区分**：

- **Pearl 因果**：在约束 $\mathcal{C}$ **内部**做因果推理（"这只鸡是否导致了这个蛋"）
- **CET 因果**：问约束 $\mathcal{C}$ **本身**的问题（"什么维持了鸡→蛋→鸡这个闭环"）

**Paper 1 的因果论证分布在两个层级**：

| 论证 | 层级 | Pearl 还是 CET |
|-----|------|:------------:|
| "action 影响 obs[0]"（spike test）| 局部（$\mathcal{C}$ 内单步）| **Pearl**（$do$ 算子有效）|
| "系统内化了因果结构"（recovery test）| 局部（$\mathcal{C}$ 内多步）| Pearl 扩展 |
| **"self 表征是自我维持的"（self-maintenance）** | **全局（$\mathcal{C}$ 本身）** | **CET**（选择效应）|
| **"发育顺序不可颠倒"（async awakening）** | **全局（$\mathcal{C}$ 本身）** | **CET**（identifying assumption）|

**这修正了之前手册里 Pearl 分层的一个混淆**：我之前把 spike/recovery/self-maintenance 都归为 "Pearl Layer 2 干预"——从 Pearl 视角看是对的（都是 $do$ 操作），但从 CET 视角看 **self-maintenance 已经跨到全局层级**——它问的不是"这个 $do$ 会怎样"，而是"什么维持了闭环持续运转"。**这是 selection effect 的证据，不是普通干预实验**。

### 12.4 A vs B ambiguity 的 CET 精确重写

前面讨论 exp2 遗留的 A vs B ambiguity（**A**：表征分解；**B**：保留 self-correlated 变量），用 CET §9.8 的语言可以精确化：

| 立场 | CET 语言 | h 结构 |
|-----|--------|--------|
| **A（表征分解）** | 系统**同时吸收** $D_{KL}^{obs}$ 和 $D_{KL}^{int}$——相关侧面 + 因果侧面都被压低 | h 结构上体现"self vs world"分解 |
| **B（相关变量）** | 系统**只吸收** $D_{KL}^{obs}$——相关侧面被压低，因果侧面 unclear | h 只是有个和 self correlated 的可预测变量 |

**exp2 Recovery test 的困境**（用 CET 语言精确）：

- Recovery test 是 $D_{KL}^{int}$ 探测——但探测的是 **obs 层（世界端）**，不是 h 层
- 所以 recovery 高只能说"**外部世界层面**表现出可分解性"
- **不能直接说"h 内部结构是可分解的"**

**要真正区分 A/B**，需要**在 h 层面做 $D_{KL}^{int}$ 探测**——干预 h 里"疑似 self 编码"的维度，看下游预测响应。

- **exp3 的 probe test 是观测式**（从 detached h 读 label）→ 相关侧面
- **exp6 的 counterfactual spike 是干预式但在 obs 端**（喂 pred_A 假 action）→ 部分 h 层 $D_{KL}^{int}$

**真正区分 A/B 的 h 层 $D_{KL}^{int}$ 探测在 Paper 1 里没有**——这是未来实验的方向（**可能的 exp7+：直接在 h 上做 do 操作**）。

### 12.5 exp2 严格边界的 CET 精确表述

远端同步过来的 [.claude-notes/project_exp2_strict_boundary.md](../.claude-notes/project_exp2_strict_boundary.md) 已经指出 exp2 只 licenses "Implicit Causal Utilization"，不 licenses "Self-World Decomposition"。用 CET 精确化：

**exp2 严格 claim**：
- 系统的 $M$ 通过 **$D_{KL}^{obs}$ 吸收了因果通路上的相关性**
- Recovery test 提供了 **obs 层 $D_{KL}^{int}$ 的证据**（世界端 $do$ 后系统能重构 signal）
- **h 内部结构是否 A（可分解表征）还是 B（相关变量）**，recovery 无法唯一确定

**Paper 1 的完整"Self"claim 需要五个证据合取**（这是完整的严格论证）：

1. **$D_{KL}^{obs}$ 吸收**（exp2 pred gap）——相关侧面
2. **$D_{KL}^{int}$ 吸收**（exp2 recovery + spike）——因果侧面（obs 层）
3. **h 内部可读表征**（exp4 probe recall）——把 A 从 B 中分离
4. **表征功能自我维持**（exp4b self-maintenance）——CET 全局层的选择效应
5. **发育时序满足**（exp5 async）——identifying assumption

**四个条件都是 §13.8 意义上"信息通路 + 优化压力"合取**——没有任何单一实验能完成。

### 12.6 CET 视角下需要修正的前面章节表述

带着 CET 视角回看，手册里几处需要收紧：

**误读 15**（"encoding gap 不是核心新发现"）**结论正确，但理由需要收紧**：

- 前面说：encoding gap 是 ML 常识（distributed representation）
- **CET 更精确的说法**：encoding gap 是 **§13.8 第二条件（架构通路）缺失时的必然现象**——Paper 1 的贡献不是"发现 gap"，是"证明这个理论预测在最小人工系统里可复现，且量化了最小充分通路（1D trace）"

**误读 14**（"Pearl 分层 = 证据链分层"）**方向正确，需要补充**：

- 前面说：Pearl 分层是查询类型，证据链分层是 claim 类型
- **CET 补充**：**证据链的"功能"层（Level 3, self-maintenance）已经跨越到 CET 的全局因果层级**，不再是纯 Pearl 意义上的 Layer 2 干预——它测的是"约束本身持续存在的条件"，是选择效应

**"能用 ≠ 有表征"的诚实定位**（之前讨论）**用 CET 更精确**：

- 前面说：ML 视角看是常识
- **CET 精确表述**：**"能用" = $M$ 吸收了 $D_{KL}^{obs}$；"有表征" = $M$ 吸收了 $D_{KL}^{int}$ AND h 里有可读维度**——两者的分离是 §9.8 两把尺的理论预测，不是 empirical 惊讶

### 12.7 CET 框架下 Paper 1 的最终定位

**Paper 1 = §13.8 (条件集扩张) 的最小 empirical demonstration**，具体贡献：

1. **证明 §13.8 的两条件（信息价值 + 架构通路）在人工系统中可复现**
2. **量化 Level 1→2、Level 2→3 各自的最小充分架构通路**（causal loop、trace）
3. **给出发育时序作为 identifying assumption 的证据**（exp5）
4. **用 selection effect（exp4b）验证 §14.3 关于 Self 持续存在的非目的论解释**
5. **用 §9.8 的两把尺同时验证约束的相关与因果两个侧面**（spike + recovery + counterfactual）

**Paper 1 不涉及**：

- CET §15 之后的 Value / Goal / Preference（未实现的 Level 4+）
- CET §11 的约束创造类型（Paper 1 stays at Level 1-3）
- CET §16 的 $\mathcal{V}$ 层级结构（未考虑多层 viable set）

### 12.8 未来实验方向（CET 框架驱动）

从 CET 视角，几个自然的 exp7+ 方向：

1. **h 层 $D_{KL}^{int}$ 探测**：直接在 h 上做 do 操作以区分 A vs B
2. **Level 3 → 4 的架构通路**：递归自我模型 $M(M)$ 需要什么额外通路
3. **约束创造**（§11）：让 agent 不只是利用约束，还能创造持久的新约束（超越 Paper 1 的 Constraint Utilization）
4. **多层 $\mathcal{V}$**（§16）：agent 认同哪一层 viable set 决定 Goal——这需要新的实验设计

**exp7 (metacog_level4) 是这条路的第一步尝试**（虽然目前 paused）——目标就是探测 Level 3→4 的条件集扩张。

---

**总结**：**用 CET 重读 Paper 1，最重要的收获是把"encoding gap 是新发现"这个过强的表述**，替换为**"encoding gap 是 §13.8 第二条件缺失的可预测现象——Paper 1 的原创贡献是最小充分构造"**。这个转换让 Paper 1 从"一个关于 self 的实验报告"提升为"CET 理论框架的最小可验证测试"——更谦逊也更强。
