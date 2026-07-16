# exp4b：Self-Maintenance 深读

**用途**：单独深读 Paper 1 中 exp4b (Self-Maintenance)。**这个实验之前作为 exp2 三层证据链的 Layer 3 被引用过，但它本身有独立的方法论和 claim 值得深挖**。

**前置阅读**：先看 [exp3_exp4_encoding_gap.md](exp3_exp4_encoding_gap.md) 建立"编码通道"的概念，再读这份文档。

---

## 目录

1. [前置：为什么需要 exp4b](#1-前置为什么需要-exp4b)
2. [exp4b 架构详解](#2-exp4b-架构详解)
3. [三阶段训练结构](#3-三阶段训练结构)
4. [Ablation 结果与解读](#4-ablation-结果与解读)
5. [严格边界与常见误读](#5-严格边界与常见误读)
6. [Selection Effect：不是意愿，是幸存者偏差](#6-selection-effect不是意愿是幸存者偏差)
7. [附录：CET 视角](#7-附录cet-视角)

---

## 1. 前置：为什么需要 exp4b

### 承 exp2/3/4 而来的问题

前面几个实验已经建立：
- **exp2**：系统能利用因果 loop 做预测（implicit causal utilization）
- **exp3**：h 里没有可读的"我在动"表征（12% recall）
- **exp4**：加一维 trace，h 里出现可读表征（60%+ recall）

**但一个 subtle 的问题还没被回答**：**exp4 里 h 里出现的"可读表征"，是被 aux head 硬塞进去的假象，还是系统真正内化的表征？**

### Aux head 的双面性

**Aux head 的作用**：在 Phase 2 里持续施压 h 编码 self 状态。
- **好处**：给了 h 一个"目标"，让线性 probe 能测出效果
- **风险**：**表征可能只是"aux loss 逼出来的临时结构"**——一旦撤走 aux，表征立刻崩溃

**这两种情况对"自我表征"的哲学 claim 意义完全不同**：

| 情况 | 表征性质 | 哲学含义 |
|------|---------|---------|
| **表征依赖 aux 支撑** | 外部监督的假象 | 系统本身没内化，只是被逼着"演" |
| **表征在无 aux 时依然存在** | 系统动力学内化 | 表征是**功能性**的，系统真正"用"它 |

**exp4b 就是设计来区分这两种情况的**：**先用 aux 训练出表征，然后撤走 aux 看它能不能自己保持**。

### 关键设计原则：Ablation as Necessity Test

**Ablation（消融）不是为了破坏，是为了测试必要性**：
- 如果 aux 是**必要**支撑，撤走后表征崩溃 → 表征是"演出来的"
- 如果 aux 只是**训练时的脚手架**，撤走后表征依然存在 → 表征是**内化的**

**exp4b 的对照设计**：Causal vs Control，**只有 Causal 组的表征是内化的**。这个不对称性就是"self-maintenance"的证据。

---

## 2. exp4b 架构详解

**文件**：[experiments/exp4b_self_maintenance.py](../../experiments/exp4b_self_maintenance.py)

### 与 exp3/exp4 的关键差异

**⚠ 注意**：exp4b **不是** exp4 的延续，**是 exp2 setup 的延续**（3 通道 + 无 trace）。

| | exp3/exp4 | **exp4b** |
|---|---------|-------|
| **信号通道数** | 4d | **3d** |
| **有 trace 吗** | exp3 无, exp4 有 | **无 trace**（GRU 输入 = 3d obs） |
| **Action 来源** | 外部随机 uniform(-2,2) | **W_action(h) 生成**（但 W_action 是**固定 buffer**） |
| **W_action 训练吗** | 无 W_action | **固定** buffer（**未训练**，稀疏 motor 掩码） |
| **Aux head 分类** | 三分类 (active/trailing/quiet) | **二分类** (is burst_active) |
| **训练阶段** | Phase 1 + Phase 2 | Phase 1 + Phase 2 + **Phase 3 Ablation** |

**核心架构**（[exp4b:155-184](../../experiments/exp4b_self_maintenance.py#L155)）：

```python
class Model(nn.Module):
    def __init__(self, c):
        self.gru = nn.GRUCell(nc, hs)              # 3d obs → 192 hidden
        self.pred_sensory = nn.Linear(hs, nc)      # 感知重建头
        
        # 稀疏 motor 掩码：只有 25% 的 h 维度参与 action 生成
        n_motor = int(hs * c["motor_frac"])        # motor_frac = 0.25
        W_act = torch.randn(hs, 1) * 0.5           # 随机权重
        mask = torch.zeros(hs, 1)
        motor_idx = torch.randperm(hs)[:n_motor]
        mask[motor_idx] = 1.0                       # 只有 motor 位置为 1
        gain = math.sqrt(hs / n_motor)              # 补偿稀疏性的增益
        self.register_buffer("W_action", W_act * mask * gain)  # ← 固定 buffer
```

**注意 `register_buffer`**：**W_action 是 buffer，不是 Parameter**——**从架构上就无法被训练**。

### 稀疏 motor 掩码的意义

**只有 48/192 = 25% 的 h 维度直接影响 action**——**其余 75% 的 h 维度不参与 causal loop**。

**这个设计模拟生物系统**：**大脑里只有一小部分神经元是运动神经元**（motor neurons），大部分是其他类型（感觉、联合、抑制等）。**exp4b 的稀疏 motor 就是这个抽象的最小实现**。

**关键效果**：**W_action 只从 motor subset 读**——**Causal loop 的信息通路只经过 h 的这 25% 维度**。**其他 75% 维度不受 causal loop 影响**。

### 两种 diagnostic heads

**BinaryHead (S1a)**（[exp4b:124-131](../../experiments/exp4b_self_maintenance.py#L124)）：
```python
class BinaryHead(nn.Module):
    """S1a: bare linear readout of h_multi."""
    def __init__(self, hidden_size):
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, h):
        return self.fc(h).squeeze(-1)
```

**GatedBinaryHead (S1b)**（[exp4b:134-149](../../experiments/exp4b_self_maintenance.py#L134)）：
```python
class GatedBinaryHead(nn.Module):
    """S1b: action-trace gated readout of h_multi."""
    def __init__(self, hidden_size, mid_dim=64):
        self.gate = nn.Sequential(
            nn.Linear(hidden_size + 1, mid_dim),  # 拿 h + action_mag
            nn.Tanh(),
            nn.Linear(mid_dim, hidden_size),
            nn.Sigmoid(),
        )
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, h, action_mag):
        gate_in = torch.cat([h, action_mag], dim=-1)
        g = self.gate(gate_in)
        return self.fc(g * h).squeeze(-1)          # gated h → linear readout
```

**两者的区别**：
- **S1a（bare）**：纯线性 readout——**测 h 里独立是否有可读表征**
- **S1b（gated）**：**gate 由 h + action_magnitude 决定** → 相当于"用 action 门控 h 后再读"——**测 h 里的信息是否可以借助 action 上下文提取**

**关键论证结构**：**Paper 1 主要报告的是 S1a**（bare）——因为这是**最严格**的测试（如果 h 里表征存在，bare linear probe 就能读出，不需要 gate 帮忙）。

---

## 3. 三阶段训练结构

### Phase 1 (100K 步)：Reservoir 感知训练

**训练模式**（[exp4b:377-384](../../experiments/exp4b_self_maintenance.py#L377)）：
```python
if is_phase1:
    sens_pred = model.step(xt_sens)                # step 用 detach（reservoir）
    loss_sens = nn.functional.mse_loss(sens_pred, xt_sens)
    opt1.zero_grad(); loss_sens.backward(); opt1.step()
```

**特点**：
- **只训 pred_sensory + GRU + EMA**（W_action 是 buffer，冻结）
- **Action 从第 1 步就开始施加**（不像 exp5 async——exp4b 的 W_action 从头就在用，只是不训练）
- **没有 aux head 参与**
- **h 学到稳定 attractor**，pred 学到重建

**注意**：这里 `step` 方法内部 h_multi 是 detach 的（reservoir 模式），所以 Phase 1 里 GRU 的梯度**来自**`pred_sensory` 的反向传播——这和 exp3/4 Phase 1 一致。

### Phase 2 (80K 步)：Aux 施压 + 双 probe 训练

**训练模式**（[exp4b:389-411](../../experiments/exp4b_self_maintenance.py#L389)）：
```python
elif is_phase2:
    sens_pred, h_multi_live = model.step_live(xt_sens)    # h_live 不 detach
    loss_sens = nn.functional.mse_loss(sens_pred, xt_sens)
    
    target = torch.tensor([1.0 if burst_active else 0.0])
    logit_bare = bare_head(h_multi_live)                   # bare probe 训练
    loss_aux = aux_weight * bce(logit_bare, target)        # 通过 h_live 施压 h
    
    total_loss = loss_sens + loss_aux                       # 组合 loss
    opt2.zero_grad(); total_loss.backward(); opt2.step()
    
    # Gated probe 单独训练（不影响 GRU）
    logit_gated = gated_head(model.h_multi.detach(), ...)
    loss_gated = bce(logit_gated, target)
    opt_gated.zero_grad(); loss_gated.backward(); opt_gated.step()
```

**关键设计**：
- **`bare_head` 通过 h_live 施压 GRU**（loss_aux 反向到 GRU）→ **这是塑造 h 的力量**
- **`gated_head` 从 detached h 读**（梯度不回流 h）→ **只是 diagnostic，不影响 h**

**注意**：**Bare head 兼职两个角色**——**训练时作为 aux head 施压 h**，**测量时作为 probe 读 h**。这与 exp3/4 里 aux 和 probe 分开是**不同**的架构选择。

**为什么这样设计**：**S1a 是 bare head，能通过施压学到多少，就代表'h 里能形成的最简单的可读结构'**。这个 "施压 + 测量"合一的设计**最大化了表征的机会**——**如果连 bare head 施压都塑不出 h 的表征，那 h 就真的没这个结构**。

### Phase 3 (5K 步)：Ablation——所有 probe 冻结

**训练模式**（[exp4b:471-482](../../experiments/exp4b_self_maintenance.py#L471)）：
```python
else:  # is_ablation
    with torch.no_grad():                                   # 全 no_grad
        model.step(xt_sens)                                 # 只更新 h_multi 状态
        
        logit_bare = bare_head(model.h_multi)               # 只读，不训
        p_bare = torch.sigmoid(logit_bare).item()
        pred_bare = 1 if p_bare > 0.5 else 0
        actual = 1 if burst_active else 0
        bare_ok = int(pred_bare == actual)
    
    ablation_bare_window.append(bare_ok)
    ablation_acc.append(bare_ok)                            # 累计 accuracy
```

**关键特点**：
- **`with torch.no_grad()`**：全部 forward，无 backward
- **所有权重冻结**：pred_sensory / GRU / bare_head / gated_head 都不动
- **只有 h_multi 状态在通过 GRU 前向被更新**（跟着 obs 走）
- **5000 步的 accuracy 累计**就是"self-maintenance retention"

**"撤走 aux"的严格含义**：**不是权重被清零**，是**梯度停止流入 aux head 和 GRU**——**它们保持 Phase 2 结束时的最佳权重，但不再被优化**。

---

## 4. Ablation 结果与解读

### 关键数字

**Causal 组**：Phase 2 结束时 S1a ≈ 95.3%，Ablation 5000 步后 **≈ 94.9%**（几乎不变）
**Control 组**：Phase 2 结束时 S1a ≈ 91.9%，Ablation 5000 步后 **≈ 53.9%**（塌到接近 chance）

**两组 gap 巨大**：**41.0 pp**（percentage points）——这比 exp2 recovery test 的 17.6 pp gap 大得多。

### 为什么 Causal 保持而 Control 崩溃

**Causal 组的 h 参与 causal loop**：
- `action = h @ W_action` → action 影响 obs → obs 更新 h
- **h 里的 motor subset**（25% 维度）**直接决定 action**
- **burst_active 状态影响 action 是否被应用（gamma_eff）**
- **h 需要持续追踪 burst_active 才能预测好 obs**
- **撤走 aux 后，pred_sensory 的 MSE 依然要求 h 追踪这个信息**
- **selection effect**：**功能上必要的表征被 pred_sensory 的 MSE 隐式维持**

**Control 组的 h 不参与 causal loop**：
- action 是外部 AR(1) 噪声，与 h 无关
- **h 里追踪 burst_active 对预测 obs 没帮助**（burst_active 与 action 无关联到 h）
- **只有 aux head 的 loss 在推 h 编码 burst_active**
- **撤走 aux 后**，pred_sensory 的 MSE **不 caring** burst_active
- **h 里的 burst_active 编码逐渐衰减/被覆盖**——**没有功能价值，就没人维持**

**核心机制**：**在 Causal 组，追踪 burst_active 是"预测好的副产品"；在 Control 组，追踪 burst_active 是"外部监督的额外任务"**。**撤走外部监督后，只有前者能靠 pred_sensory 的 MSE 维持**。

### 为什么这个 gap 比 exp2 recovery 大

**exp2 recovery gap**：Causal 74.8% vs Control 57.2%（gap = 17.6 pp）
**exp4b ablation gap**：Causal 94.9% vs Control 53.9%（gap = 41.0 pp）

**为什么 exp4b 的差异更 dramatic**：

**exp2 recovery 测的是"h 状态适应新分布的能力"**——两组都能适应，只是速度不同（一个快一个慢）。

**exp4b ablation 测的是"表征在无监督时的保留"**——**Causal 组有功能锚**（pred 的 MSE 需要 burst_active 信息），**Control 组完全没有锚**（外部监督撤了就没别的支撑）。**这是"有锚 vs 无锚"的差异，比"快锚 vs 慢锚"更 dramatic**。

**这也是为什么 exp4b 是 exp2 三层证据链里 Layer 3（功能层）的证据**——**它测的是"表征是否有功能价值"，不是"表征是否能形成"或"表征是否结构化"**。

---

## 5. 严格边界与常见误读

### exp4b 严格证明了什么

1. **Causal loop 存在时，特定表征（burst_active）在无 aux 支撑下可以自我维持**
2. **Control（无 causal loop）里，同样的表征在无 aux 时衰减**
3. **两者差异归因于 causal loop 的功能价值**（因为其他条件相同）

### exp4b **不能**证明什么

1. **不能证明 h 里有"自我意识"** —— 只证明 burst_active 这个具体的、有明确功能的表征被维持
2. **不能证明维持是"有意"的** —— 是 selection effect（幸存者偏差），不是"想要"维持
3. **不能证明所有 self-related 表征都自维持** —— 只测了 burst_active 一种
4. **不能证明长期稳定性** —— 只测了 5000 步，未来是否衰减未知
5. **不能证明可跨 setup 迁移** —— 只在 3 通道 + specific hyperparams 下测过

### 常见误读

**误读 1**："exp4b 证明系统'不想忘记'自己"
- **错**。**是幸存者偏差**——**功能价值让维持成为副产品**，不是"意愿"。Control 组不是"想忘"，是"没功能所以自然衰减"。

**误读 2**："exp4b 证明 self 表征是 permanent 的"
- **错**。**只测了 5000 步**。**长期稳定性未测**。可能过更长时间（如 50K 步）Causal 组也会漂移。

**误读 3**："exp4b 里的 W_action 训练学到了 causal loop 结构"
- **错**。**W_action 是 fixed buffer**（`register_buffer`），**从头到尾未被训练**。**causal loop 的结构是通过 pred_sensory 的 MSE + W_action 的固定映射**被系统利用的——**不是**通过训练 W_action 学出来的。

**误读 4**："Ablation phase 里 h 也冻结了"
- **错**。**Ablation 只是权重冻结**（no_grad），**h_multi 的状态每步都在通过 GRU 前向被更新**（因为 obs 每步都在变）。**h 的"状态动力学"在 Ablation 里依然活着，只是不再有梯度塑造它**。

**误读 5**："exp4b 用了 aux head，所以它测的还是'监督下的表征'"
- **半错**。**Phase 2 里确实用 aux 训练**——但 **Ablation phase 完全无 aux**。**Ablation 里测的是"撤走监督后的保留"**——这才是 self-maintenance 的关键测量。

**误读 6**："S1a 和 S1b 是两个不同的 probe，可以互换"
- **错**。**S1a 是 bare linear probe**（严格线性）——**Paper 1 主要报告 S1a**。**S1b 是 gated 版本**（有非线性 gate）——**只作为 diagnostic**，不作为核心 claim。**因为如果 self-maintenance 只在 S1b 上成立、S1a 上失败，就说明"表征需要 action gate 才能读"——那不是"独立可读表征"**。

---

## 6. Selection Effect：不是意愿，是幸存者偏差

**这是 exp4b 最容易被 oversell 的哲学含义**——**必须清晰地区分"选择效应"和"意愿"**。

### Selection Effect 的严格含义

**逻辑非常朴素**：
- 表征 A 对预测 obs **有用** → pred_sensory 的 MSE 会**隐式推动** h 维持 A → **A 被看到**
- 表征 B 对预测 obs **无用** → pred_sensory 的 MSE **不管** B → **B 衰减** → **B 消失**
- **你能观察到的所有"自维持"的表征，都是"恰好对预测有用"的表征**

**这是"事后解释"，不是"预测性论证"**：
- **不告诉你**："这个表征将会自维持"（need to know functional value first）
- **只告诉你**："自维持的表征必然有功能价值"（能观察到就意味着幸存下来）

### 对比：常见的"意愿"误解

**错误的类比**：**"系统想要保持自我，所以维持了 self 表征"**
- 这是**目的论**（teleology）—— **将功能归因于意图**
- **exp4b 没有证明任何"意图"**——只证明了功能上必要的表征被 selection 保留

**正确的类比**：**"进化中被保留的生物特征都是有生存价值的"**
- 这是**幸存者偏差**——**看不到"想保留但没生存价值"的特征，因为它们没留下**
- **exp4b 里 Control 组的表征就是"没生存价值的特征"**——**它衰减不是因为"想忘"，是因为"没功能"**

### 为什么这个区分重要

**如果误读成"系统想要保持自我"**：
- 会走向**意识/意图**的强 claim
- **exp4b 不支持这个 claim**——只支持功能主义解读

**如果正确理解为"selection effect"**：
- **保持谦逊定位**——**表征持续 = 功能持续，不是意愿持续**
- **可以推广到进化生物学、经济学、社会学**里的所有 selection 论证
- **不做超出证据的哲学 leap**

### 与 CET §14.3 的对应

**CET §14.3 明确说**：**"Self 持续存在不是'意愿'，而是选择效应"**——**exp4b 是这个 CET 命题的最小 empirical demonstration**。

**逻辑链**：
```
CET §14.3 断言：Self 持续存在 = selection effect（不是意图）
    ↓
exp4b empirically 显示：功能有用的表征自维持，无功能表征衰减
    ↓
两者本质是同一个断言在不同层次的表达
```

**这个映射让 exp4b 在 CET 框架里的位置非常清晰**——**它不是关于"self 意识"的实验，是关于"什么条件下功能性结构被自然选择保留"的实验**。

---

## 7. 附录：CET 视角

### exp4b 在 CET 的层级里

**回顾 CET 层级**（从 CET §6.4-6.5）：
- **Pearl 局部因果**：在约束 C 内部做单步干预推理
- **CET 全局因果**：问约束 C 本身的持续条件

**exp4b 是 CET 全局的实验**：
- 不是问"某次干预会怎样"（Pearl 局部）
- 是问"什么让这个 causal loop 结构持续存在"（CET 全局）
- **selection effect** = **CET 全局层的机制**

**这是为什么之前把 exp4b 归为"Pearl Layer 2 干预"是不准确的**——**它更接近 §14.3 里 selection effect 的实验**。

### exp4b 在 §13.8 视角下

**§13.8 条件集扩张的两条件**：
1. 理论必要性：I(X; S_{t+1} | conditions) > 0
2. 架构必要性：信息通路存在

**exp4b 验证了一个隐含的第三条件（尚未在 §13.8 中显式化）**：
- **功能价值条件**：**扩张后的表征需要在系统持续运行中有功能价值**，否则会衰减
- **exp4b 的 Control 组失败**：即使前两条件都满足（信息价值 + 通路），如果表征无功能，也不能持久

**这可能是 CET §13.8 需要补充的第三条件**（记在 `papers/cet_updates_pending.md` 里的 open item）。

### exp4b 揭示的表征生命周期

**完整的表征生命周期**（推断自 exp4b + exp4）：
1. **形成阶段**：架构提供通路 + aux 施压 → 表征被建立
2. **维持阶段**：表征需要有 functional value → 被 pred 的 MSE 隐式维持
3. **衰减阶段**：无 functional value → 表征被其他信号覆盖

**Paper 1 通过 exp4 测阶段 1，通过 exp4b 测阶段 2 和 3**——**完整覆盖了表征生命周期的三个阶段**。

---

## 学习检查清单

读完这份文档后，你应该能回答以下问题：

- [ ] exp4b 承接 exp2/3/4 引出的什么关键问题？
- [ ] 为什么 aux head 的双面性使得 self-maintenance 测试必要？
- [ ] exp4b 的架构和 exp3/exp4 有哪些关键差异？（3 通道、W_action、稀疏 motor 掩码等）
- [ ] W_action 是 buffer 而非 parameter 的意义是什么？
- [ ] Phase 1 / Phase 2 / Phase 3 分别做什么？训练什么？
- [ ] 稀疏 motor 掩码（25% 神经元参与 action）的意义是什么？
- [ ] S1a (bare) 和 S1b (gated) 的区别？为什么 Paper 1 主要报告 S1a？
- [ ] Ablation phase 里"权重冻结"和"h 状态更新"如何共存？
- [ ] Causal 94.9% vs Control 53.9% 的机制解释是什么？
- [ ] 为什么 exp4b 的 gap (41pp) 比 exp2 recovery 的 gap (17.6pp) 更 dramatic？
- [ ] "Selection effect" 和 "意愿"的区别是什么？
- [ ] exp4b 严格证明了什么？没能证明什么？
- [ ] exp4b 在 Paper 1 五个实验合力证明 self-world decomposition 的证据链里扮演什么角色？
- [ ] exp4b 与 CET §14.3 的对应关系？

---

**下一步（读完这份文档后可选路径）**：

1. **exp5（Async Awakening）深读**——理解时序作为 identifying assumption
2. **exp6（Agency Gain）深读**——理解 Pearl Layer 3 反事实 + 定量测量
3. **回过头重读 exp3+exp4 深读的 §4.2a canonical 三段总结**——现在可以用 exp4b 的视角检查这个总结是否还需要精细化
4. **想跑 experiment**——用 `--quick` mode 跑一次 exp4b 看数字
5. **回到 papers/three_levels/notes.md 更新**——用 exp4b 的 "selection effect" 视角完善 L2 vs L3 的区分（L2 已经有 selection effect，L3 是什么？）
