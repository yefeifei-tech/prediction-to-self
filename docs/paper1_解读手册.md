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
8. [五指标 vs 八指标：为什么不能压缩](#五指标-vs-八指标为什么不能压缩)
9. [关键数字速查表](#关键数字速查表)
10. [常见误读与澄清](#常见误读与澄清)
11. [代码结构导航](#代码结构导航)

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

$$\text{spike\_ratio}[\text{ch}] = \frac{\text{Err}_{\text{disconnected}}[\text{ch}]}{\text{Err}_{\text{connected}}[\text{ch}]}$$

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

$$\text{recovery} = \max\left(0,\ 1 - \frac{\text{mean}(E_{\text{last 100}})}{\max(E_{\text{first 50}})}\right)$$

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

**关于测试期状态**：所有 spike test / recovery / ablation **全程在 `torch.no_grad()` 里**——权重完全冻结。recovery 测的是 **h 状态动力学**的适应，不是权重学习。

**关于数字复现**：v0.3.2 setup（sparse motor + action delay + peak-of-5 spike）能测出 spike ~13.8x；当前 released `experiments/exp2_causal.py` 用简化的 mean-over-2000 spike 测量，spike ~1x。**测的东西不同**：v0.3.2 测**瞬时冲击**，当前 code 测**稳态差异**。**Recovery 数字（74.8% vs 57.2%）在两个版本里都可复现**，是 Layer 2 的稳定证据。

**关键代码**：
- 训练：[exp2_causal.py:87](../experiments/exp2_causal.py#L87) `train_model()`
- Spike 测试：[exp2_causal.py:143](../experiments/exp2_causal.py#L143) `spike_test()`
- 长断连：[exp2_causal.py:202](../experiments/exp2_causal.py#L202) `long_disconnect_test()`
- Self-maintenance：[exp4b_self_maintenance.py](../experiments/exp4b_self_maintenance.py)

**在证据链中的位置**：**步骤 2 + 3 + 6**（Pearl Layer 2 干预 + 混淆控制 + self-maintenance 功能验证——三层加起来是 exp2 完整证据）

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

**如果误以为 spike 应该"只 Causal 有、Control 没有"**，就会得出"exp2 不成立"的错误结论。**这是把 Layer 1（存在性）和 Layer 2（因果性质）的角色搞混了**。

**正确的分层理解**：
- **Spike test**（Layer 1）：证明两组都学到了"action 影响 ch0"这个通道特异性模型（**存在性**）
- **Recovery test**（Layer 2）：证明只有 Causal 组的模型是结构化因果的，Control 只是统计关联（**性质**）
- **Self-maintenance**（Layer 3）：证明 Causal 的表征是内化的，不是 aux 的假象（**功能**）

**必须先建立 Layer 1，Layer 2 的差异才有可解读性**——否则"两组 recovery 不同"这个事实不知道是关于什么的差异。详见 [exp2 章节的分层证据链](#exp2-causal-budding--因果萌芽)。

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
