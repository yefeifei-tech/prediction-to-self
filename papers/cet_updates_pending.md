# CET Updates Pending — From Paper 1 Empirical Audit + Three-Levels Discussion

**用途**：收集所有从 Paper 1 实证审计和 three_levels 讨论中 emerge 的、可以反哺 CET 主论文的更新建议。**这是一份 pending 清单，需要 Evan 判断哪些采纳、以什么方式采纳**。

**分级说明**：
- 🟢 **安全**：可直接改（数字修正、术语精确化）
- 🟡 **表述精细化**：改动方向明确，但影响 CET 表述风格，需 Evan 判断
- 🟠 **战略性新加**：可能扩展 CET 结构，需 Evan 决定加不加、加在哪
- 🔴 **不建议进 CET**：明确不应吸收（如三层框架属于独立 paper）

---

## 🟢 安全修正

### §13.8 数字引用（Paper 1 实际数据核对）

| 原文 | 修正为 | 依据 |
|-----|-------|------|
| "Phase 1（随机行动，pred_A 未利用 action 输入）时 pred gap ≈ 0%" | "Phase 1（随机行动）pred gap **98.8%** —— 这是机械补偿（pred_A 有 action 输入即可算），不是 causal 依赖；**spike ≈ 0.95×** 说明无因果结构" | Paper 1 §6.1 |
| "Phase 2b 后 pred gap 提升到 93.7%" | "Phase 2b（trained policy）pred gap **80.7%**（比 Phase 1 反而下降 —— trained action 更结构化，pred_A 难以完美补偿）；**spike 17.32×** 才是真正区分 Phase 1 vs 2b 的指标" | Paper 1 §6.1 |
| "v0.3–v0.4 时线性探针只有 70%" | "v0.3-v0.4 时 **aggregate action-state probe（三分类 active/trailing/quiet 综合）70%** —— 但 **trailing-specific recall 只有 12.3%**（这才是 encoding gap 的真正证据）" | Paper 1 §3.3, Table 1 |
| "v0.4.13 加入本体感受通道后探针跳到 80.1%" | "v0.4.13 加入本体感受通道后 **trailing recall 从 12% 跳到 56.5%**（encoding gap 突破）；**symbol grounding BA 达到 80.1%**（另一个 metric，不要与 trailing recall 混淆）" | Paper 1 §3.4 |

### 术语确认

| 位置 | 需确认 | 现状 |
|-----|-------|------|
| CET 全称 | Constraint Emergence Theory（不是 Conditional Emergence）| ✓ |
| ORI 定位 | Observer-Relative Information，配套论文，$I_{ORI} = I_{CET}$ 同一量 | ✓ |
| $M \subset \mathcal{C}$ 语义 | 已在 Axiom 3 注中说明 = "M 的物理载体属于 C 所约束的世界"| ✓ |

---

## 🟡 表述精细化

### §13.8 subject 修正（主体澄清）

**原文（含 misleading 意味）**：
> "系统在持续降低 $I_{CET}$ 过程中，逐步将条件集扩张到包含自身状态而形成的稳定压缩结构。"

**问题**：暗示"系统主动扩张条件集"—— Paper 1 实证 empirically 显示**系统不主动扩张**：
- Action → obs 通路是**实验者预设**的
- GRU 输入维度是**实验者预设**的
- "扩张"的真实机制 = **实验者提供架构** + **系统的 readout 学会利用**

**建议修正**：
> "**当实验者/环境提供架构通路且信息条件满足时**，系统的 readout 在持续降低 $I_{CET}$ 的优化压力下发现并利用该通路，从而**'表征'在被利用的意义上涌现**。"

**理由**：Paper 1 reservoir setup（GRU 冻结、只训 readout）empirically 显示"扩张"发生在 readout 层，不发生在系统自主决定层。**主体是"环境提供 + readout 发现"，不是"系统自主扩张"**。

**影响面**：这个修正会影响 CET §13.8 的核心叙事，需要 Evan 判断是否接受"环境提供 + readout 发现"这个较弱的主体表述。

### 区分"学习 in dynamics" vs "学习 in readout"

**当前 CET 表述**：学习作为一个整体动作，没有区分发生在哪一层

**建议加入**（可以是 §13.8 的 footnote 或独立小节）：

> "学习可以发生在两个不同层级：
> - **Readout 层学习**（如 Reservoir Computing 中 pred 训练）：从已存在的 dynamics 中选择/读出任务相关子空间
> - **Dynamics 层学习**（如标准 DL 中 recurrent 权重训练）：重塑系统的内部动力学结构本身
> 
> Paper 1 的 reservoir setup 只做 readout 层学习就足以触发 Level 1→2 涌现——**表明持续动力学（Axiom 1）+ 可训练 readout 是 Level 1→2 涌现的最小架构**，dynamics 层的学习不是必要条件。"

**理由**：这个区分让 CET 对"学习"的表述从模糊变精确，且为未来"三层框架"paper 埋下引用位。

---

## 🟠 战略性新加内容

### 提议 A：§13.8 补时序条件

**Paper 1 exp5 显示**：架构齐全（4 条件通路都在）+ 信息条件满足 ≠ 涌现。**训练时序**（perception 先于 action）也是必要条件——即使 LR 调对了，同步训练也失败。

**候选加法**：

> "**§13.8a 时序条件（第三条件）**：条件集扩张不仅需要（1）信息价值 + （2）架构通路，还需要（3）**identifying temporal order** —— 条件必须以特定顺序在学习过程中被引入，dependency structure 决定顺序。Paper 1 exp5 显示 perception 必须先于 action 稳定；同步引入两者会导致 gradient interference 阻止涌现。"

**Evan 决定**：加还是不加？如果加，作为 §13.8 的子条件还是作为独立 §13.8a？

### 提议 B：§9.8 用 Paper 1 metric 做具体化示例

**当前 §9.8 抽象讨论了两把尺（$D_{KL}^{obs}$ vs $D_{KL}^{int}$）**，可以加一个 Paper 1 具体化示例：

> "**Paper 1 应用示例**：
> - **pred gap** = $D_{KL}^{obs}$（观察式：知道 action 后预测改善多少）
> - **spike test / recovery** = $D_{KL}^{int}$（干预式：$do$(断开 action)后系统响应）
> - **exp6 counterfactual spike** = h 层 $D_{KL}^{int}$（Pearl Layer 3 反事实）"

**Evan 决定**：加还是不加？作为 §9.8 里的 footnote / example box / 独立 subsection？

### 提议 C：§13.8 加 mechanism 说明——reservoir view

**Paper 1 reservoir 事实揭示了一个 mechanism-level 洞察**：信息通路的物理基础是 stateful dynamical system。**这可以在 CET 里明确出来**：

> "**信息通路的物理基础**：CET 假设 X 有'架构通路到达预测计算'，但没说这条通路的物理形态。从 Reservoir Computing 视角，最小满足这条要求的物理形态是**stateful dynamical system**（h 持续更新的循环结构）——它保证输入信号能被传播到内部状态并被 readout 提取。
> 
> **推论（未经完整验证）**：**stateless 架构**（如 pure feed-forward，无 KV cache 的 vanilla Transformer 变体）不满足这个物理形态，因此**无法通过 reservoir + readout 机制形成 self-representation**——即使其他 CET 条件都满足。"

**Evan 决定**：加还是不加？这个 claim 加强了 CET 的可测性，但也带来一个未经完整验证的 empirical 预测。

---

## 🔴 不建议进 CET

### 三层框架（Dynamics / Readout / Dynamics Shaping）

**不要塞进 CET**。理由：
- 三层是关于 **ML learning 的机制层重构**
- CET 是关于**主体涌现的哲学理论**
- 两者层级不同，混合会稀释两个都
- **正确做法**：CET 在 related work 里**引用**三层框架 paper（当它写出来后），不吸收进正文

**CET 里如果一定要提三层，只能是 footnote**：
> "For a mechanism-level view of how L1-L3 learning modes correspond to CET's condition set expansion, see [Ye 2027+ Three Levels paper]."

---

## 推荐执行顺序

1. **🟢 安全修正**先做（4 处数字、术语确认）—— 无风险，直接改
2. **🟡 subject 修正**做，但保留原句作为 footnote 或 alternate wording 备选
3. **🟠 战略性加入**每一项单独找 Evan 讨论，不批量决定
4. **🔴 三层框架**明确不进 CET，等独立 paper 发出后 CET 只引用

---

## Cross-references

- 完整讨论上下文：`.claude-notes/project_paper1_as_cet_demonstration.md`
- Paper 1 empirical audit 细节：`docs/paper1_解读手册.md` §12（用 CET 重读 Paper 1）
- 三层框架 paper 工作目录：`papers/three_levels/`
