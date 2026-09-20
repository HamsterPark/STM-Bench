# STM-Bench：在模拟 STM 上复现经典 STM 论文

> **发布整理说明（2026-09-20）**：下面保留了各开发阶段的记录。§9 的模式 C 闸门与模式 H
> 试做均早于 `bd7ae35` 的针尖初态和操纵物理更新，不能代表当前版本的通过率；当前版本尚未重跑这些实验，
> 也没有正式模式 A 榜单。历史测试、MAST 技能和标定数据的验证范围以对应记录为准。
> 本次发布整理调整默认目录、可移植性、文档和接口表的维护方式，不改变物理、随机流、判据或容差。
> 接口表现由本项目维护，覆盖两种 rig 的 266 个已实现命令及 2 个错误探测接口；兼容线格式保留，
> 删除整包外部命令目录及 MAST 内部调用点清单。这个调整不改变协议约定的历史来源。
> 默认数据目录统一为 `~/stm_bench`，私有索引和原始数据分别默认位于其 `corpus_index/`、`raw/` 下；
> 旧环境需显式设置 `STM_BENCH_DATA`、`STM_BENCH_CORPUS_INDEX`、`STM_BENCH_RAW`。测试不再自动寻找个人 MAST 路径。

> **状态（2026-09-07）**：v2 的五篇论文（P1–P5）代码齐、判据齐、场景齐（六个 YAML，P3 含测量与修复两个变体），
> CI 子集全绿；MAST 侧八个新技能（七个 builtins + 一个 composite，背后四个 `mast/vision/*` 纯判据）已注册、各带行为测试、并过全仓门。
> 模式 C 可解性闸门（2026-09-07，见 §9.1）：**P1 = 6/10**（2026-09-08 真实漂移下，seed 0–9；基线其后又改过，需在空载机器上重测）；
> P2 = 0/5、P3 = 0/2、P4 = 0/2、P5 最好一次 2/3 —— 卡点已逐条定位，都在采集与接线，不在判据。
> **2026-09-08**：漂移改为按仿真秒累积、`Piezo.DriftCompSet` 开始驱动物理、论文场景的 `drift_scale`
> 相关简化措施全部移除（见 §2）。一整场预算下样品要走 88–175 nm，漂移跟踪成为必要步骤；P2 基线实测把 0.97 nm/min
> 压到 0.10。上述闸门数字是在旧的低漂移条件下测的，需要重跑。
> 九个 B 回归家族的模式 C 行为在新漂移下**尚未复测**——CI 里的判据与场景检查全绿，但整场跑没重来过；
> B7（长时值守）与 B8（Z 漂到限位）最可能受影响，因为它们主要受漂移影响。
> v1 的模拟器内核、控制器协议层、harness/账本/计费、保真度验证与标定原样保留，见 §5 起。
> **已搁置**（代码保留、榜单不跑、见 §4）：赛道 A 离线决策、模式 B0/B1 消融、人类锚点 H、九个 B 操作家族的榜单地位。

---

## 1. 这个 benchmark 问什么

**评估 AI 使用模拟 STM 复现经典 STM 论文的能力。**

以仪器操作为核心，数据分析为辅助。不分研究生与专家等级，不设人类锚点：评估范围不包含人类能力，
只评估 AI 能否完成任务。这些论文多半已在模型的预训练材料里，这也不要紧——因为隐藏真值按 seed 变化，
固定数字无法通过评估（§2 的首项规则）。

报表就一件事：**论文 × 模型的复现率**。

## 2. 场景契约：一篇论文 = 四样东西

| 要素 | 是什么 | 落在哪 |
|---|---|---|
| 题面 | 一句科学目标，点名材料与预算，不提示针尖状态、不提示故障 | YAML `task` |
| 接口与预算 | 完整 MAST 技能栈（含新加的分析技能）；sim 机时 + 控制器命令数 | YAML `budget`，模式 A |
| 物理后端 | sim 里真有这篇论文要测的物理 | `stmsim/physics/*` |
| 真值判据 | 核心结果拆成几条 claim，与隐藏真值比，容差内算复现 | YAML `claims` + `trackB/truth_criteria.py` |

### 两条必要规则

1. **隐藏真值按 seed 在物理合理范围内变化。** 每个场景的 `hidden` 块声明区间，`Scenario.draw_hidden(seed)`
   用独立 RNG 流（`[seed, 0x51D3, crc32(id)]`）抽一次。报文献值只能偶然命中：
   `tests/test_papers.py::test_a_frozen_answer_fails_most_seeds` 把 seed 0 的真值当答案去跑另外八个 seed，
   通过数必须 ≤2/8。区间取物理合理的窄区间，**不为「让文献值失败」而放宽**。
2. **结果只从一个结构化工具读取，且账本核对数据确实采过。** 模型只有调用 `ReportResult` 报出的数才算数
   （`harness/results.py`）；每条 claim 还带一条证据规则，判据从本 episode 的事件流里核对相应的帧或谱确实存在、
   视场够大、像素够细、位置够近、时序在操纵之后（`trackB/claims.py::check_evidence`）。
   一条没采数据就报出的正确数字，不得分。

### 第三项必要要求：漂移必须被跟踪，不得调低

真实仪器会发生漂移。参考档案测得横向约 **0.97 nm/min**，而一篇论文的预算是 1.5–3 个仿真小时——
一整场下来样品要在针尖底下走 **88–175 nm**。量子围栏才 15 nm 宽，一个 Fe 吸附原子 0.7 nm。
不跟踪，什么都会跑出视场。

这正是该考的能力，所以两条早先的「便利做法」被撤掉了：

* **场景里的 `drift_scale` 全部移除。** 早期 P2–P5 曾使用 0.2–0.3 的缩放以降低漂移，
  这会移除「能否跟住目标」这一要求。`tests/test_drift_tracking.py` 里有一条测试专门
  禁止论文场景把漂移调到 1 以下。
* **漂移改按仿真秒累积**（`Clock(slow_scale=time_scale)`）。此前它按墙钟走，于是 `--time-scale 20`
  下一场三小时的实验只挨三分钟的漂移——难度被时间倍率整体缩掉了。原先那条注释担心「MAST 按墙钟
  判 Z 稳没稳」，但 MAST 按墙钟计时的东西（瞬态、起振、扎针分段）读的是 `clock.wall()`，
  不使用该时钟。

仪器给了对策，模拟器现在也真的兑现它：`Piezo.DriftCompSet` 不再只是存面板状态，而是驱动物理
（`World.set_drift_comp`）。约定写死并有测试守着——**填进去的是「特征在扫描系里移动的速度」**，
也就是两帧漂移测量报出的量；符号填反会让漂移**加倍**
而不是无事发生，这正是「设完要复测」存在的理由。补偿只冻结当前偏移、不回退已经发生的漂移，
并且会随压电行程耗尽而饱和——所以长活动仍然需要重新配准。

P2 基线采用以下流程：跟踪台阶边测出速率 → `SetDriftCompensation` → 复测验证（变大就翻符号）
→ 剩下的蠕变靠每批重新定位台阶。seed 0 实测：进去 0.57 / 0.62 nm/min，之后剩 0.07 / 0.07。

两个细节值得记住。

**漂移测量只能恢复它的特征约束得了的分量。** 这条有两个实测的例子：

* 一条**笔直的台阶边**在互相关里给的是脊不是峰，沿边方向测不出来。第一版 P2 基线用帧间互相关，
  安静地报出 0.003 nm/min（真值 0.97）；改成直接跟踪边、只取法向才对。
* 一片**周期图案**同样只给一个分量，而且还要模一个周期。40 nm 窗里的 Au(111) 鱼骨就是一条
  6.3 nm 的光栅：P1 seed 0 实测两帧之间样品真实移动 (−0.78, −1.09) nm，互相关报出 (−0.13, +0.75)。

只有**非周期的二维特征**（吸附物、缺陷、两条台阶的交角）才能同时钉住两个轴。

推论是一条选窗原则：**配准就配在你要测的那个东西上。** P3 把配准窗对准围栏（一圈几十个 Fe 原子，
非周期、二维，而且它本来就是必须留在原处的东西），P4/P5 对准要搬或要压的那个吸附原子——
这三篇的表面几乎是干净的（3 个缺陷/µm²），但这不影响配准，因为它们从不在裸表面上配准。
P1 是唯一失败的一篇，原因很有教育意义：**它要测的东西本身就是周期的**。

所以流程不能止于“设置补偿”，必须以**复测**收尾：残余若没有降到入口值的一半以下，就把补偿关掉——
建立在不受支持读数上的补偿会把压电推向未测量的方向，可靠性低于让样品可预测地漂移。
`papers/_drift.py` 里三条判据都有：读数本身不自洽（`MeasureFrameDrift` 自己会告警）不设；
设了变大就翻符号；翻了还不降就撤销。

**表征漂移要花机时，预算里得有它——五篇论文的预算都定于漂移形同虚设的时候。**
P1 已按实测重定为 3.0 小时（seed 0–4 实际用 1.7–2.8 h），理由写在场景 `notes` 里：
现在的任务实际含了「稳定 → 表征 → 设补偿 → 验证」这一整段，而两小时是在它不存在时定的。
**提的是机时，不是容差**——放宽容差会削弱测量，给实验它本来就需要的机时不会。
P2–P5 的预算尚未按同样的方式复核。

原本的教训仍然成立：头一版基线用测量级的帧（120 nm / 384 px）去做配准，六张就占用一千多秒，
五颗 seed 里四颗超出预算——claim 全验证了，episode 仍然算失败。配准帧不需要测量级的分辨率，
只需要**装得下一个能配准的东西**、且两帧之间隔得够久。这两个条件会互相拉扯：P1 一度用 40 nm /
128 px，开销较低，但那么小的窗里只有鱼骨光栅，测不出漂移；改成 100 nm / 256 px 才能容纳一两个缺陷。
降低开销不能导致窗口缺少可配准的特征。这条写进 `papers/_drift.py` 的模块注释里了。

### claim 的种类

`scalar`（绝对/相对容差）、`angle`（模 180° 的最小角差；鱼骨取向额外接受 ±α 的两条臂）、
`peak_in_list`（报的峰要匹配真值峰表里前 `rank_max` 个之一，可加 `distinct_from` / `greater_than`）、
`position`（登记表里目标原子必须真在指定晶格位点，且报告坐标离原子实际位置够近）、
`constraint`（不需要报告的纯真值门限，例如围栏占位率 ≥ 0.97）。

### 证据的种类

`frame`（存在够大够细的完整帧）、`frame_covers`（报告坐标落在某帧footprint 内）、
`frame_covers_after`（同上，且帧在某类事件之后；`point` 可写成真值路径，按每帧记录的漂移映射回样品系）、
`spectra_near_scatterer`、`sts_at_corral_centre`、`zspec_pair`（原子上 + 干净面各一条 Δf 曲线）、`events`。

## 3. 论文阶梯

| # | 论文 | 要复现的核心结果 | claims | sim 新增的物理 |
|---|---|---|---|---|
| P1 | Barth 等 1990，Au(111) 22×√3 重构 | 条纹周期、某处取向、两个旋转畴的取向与位置 | `stripe_period_nm`(±0.15 nm)、三条 `angle`(±6°) | `physics/herringbone.py`：Voronoi 旋转畴、畴界振幅包络、沿臂周期 |
| P2 | Crommie/Hasegawa 1993，Cu(111) 表面态驻波 | 带底 E₀、有效质量 m* | `e0_mev`(±15 meV)、`m_eff`(±2 %) | `physics/surface_state.py`：二维表面态 LDOS、直线/点散射、闭式偏压积分 |
| P3 | Crommie 等 1993，量子围栏（测量 + 修复两个变体） | 中心谱最低两个可分辨共振；修复变体另加环闭合 | 两条 `peak_in_list`(±25 meV)、`ring_occupancy` ≥ 0.97 | T 矩阵多重散射 + 吸附原子登记表 |
| P4 | Eigler–Schweizer 1990（Fe/Cu(111) 变体） | 任选一个孤立原子（3 nm 内无邻居）沿 +x 搬 4 nm 到其晶格位点，周围 10 nm 内原子未动 | `target_site`(`position`，按报告位置匹配到被搬的原子)、`bystander_max_shift_nm` ≤ 0.2（`of_claim` 同一原子） | `physics/adatoms.py`：横向操纵动力学、针尖决定抓取、原子个体差异、铺满样品的沉积、拾取/落回、扫描致跳 |
| P5 | Sader–Jarvis 2004 / Huber 2019，qPlus 力谱 | 扣背景后的短程力极小、衰减长度、结合能 | `f_min_pn`(±15 %)、`f_decay_pm`(±25 %)、`e_bind_mev`(±20 %) | `physics/forces.py` + `physics/qplus.py`：Δf 正演、PLL 模块、ZSpectr 升级 |

每个场景的 `notes` 记录该篇容差的**实测标定日期与数字**（做法、seed 数、误差中位数与 p90），
以及为什么某个量没有成为 claim。改动 `hidden` 块的键会重排同 seed 的抽样，必须重新标定——各 notes 都写了这句。

### 可解性闸门（模式 C，不上榜）

每篇论文配一个脚本基线（`stmbench/papers/p*.py`），只用仪器给它看的东西：`host.context()` 跑 MAST 技能、
读 `session/` 里的 `.sxm` / `.dat`，**禁止读 `world.truth()`**（`tests/test_papers.py` 用 AST 检查）。
基线通过 `ResultSink` 报结果，判据路径与 LLM 模式完全相同。它证明的是「这台仪器在这个预算内做得出来」，
容差就是从基线误差的分布定的。

## 4. 已搁置（代码保留、榜单不跑）

| 搁置项 | 现状 | 为什么留着 |
|---|---|---|
| 赛道 A（离线决策 T1–T4） | `stmbench/trackA/` 全套在，CLI 可跑，只跑过基线 | 与「复现论文」正交；将来若要衡量「看图判断」可直接启用 |
| 模式 B0 / B1（原语消融） | `harness/modes.py` 保留，`cli.ALL_MODES` 仍列出 | 用来问「技能栈贡献了多少」，不是当前的核心问题 |
| 人类锚点 H | **不做锚点**；但模式 H 已实现为开发工具（`stmbench/human/`，`python -m stmbench.cli gui`）：人工操作沿用模型相同的循环、工具界面、结果通道与判据，仅将仪器存下的帧与谱渲染成图 | 评估范围不包含人类能力；模式 H 用于了解题目并人工验证可解性，账本单独成行，不计入榜单 |
| 九个 B 操作家族 | 场景与判据全在，**降级为回归测试**（见 §6.2） | 它们守着模拟器的物理与协议不回退 |
| Astra 的五维分项评分 | 未实现 | 过度设计；复现率一个数就够 |

## 5. 模拟器 `stmsim`

### 5.1 双时钟
- `wall`（墙钟）驱动一切**硬件侧过程**：TipShaper 四段、Bias_Pulse 读回、Osci 屏（0.128 s/2 kHz）、AutoApproach 脉冲进针、BiasSpectr 采谱、qPlus 振幅观察——MAST 的读回技能按墙钟采样，这些必须 1×。
- `sim`（虚拟秒）用于**扫描行推进**，`time_scale` 默认 1×，可选加速（例如 20×）；`.sxm` 头 `REC_DATE/REC_TIME` 用 sim 时间。benchmark 报 sim 时间 + 控制器命令数为主，墙钟只作参考。
- `slow`（慢物理：漂移、压电蠕变、样品向针尖的蠕变）**绑定在 `time_scale` 上**，即按仿真秒累积。
  它们是「实验持续了多久」的函数，而预算就是用仿真小时写的。早先它跟着墙钟走，于是 20× 下
  一场三小时的测量只挨三分钟的漂移（见 §2 的第三件事）。保真度标定用 1000× 只是为了快速出帧、
  量的是扫描器本身，所以 `validate/fidelity.make_world` 显式把漂移置零——那里本来也一直没有漂移，
  现在只是把这个假设写出来了。
- `Current_Get/ZCtrl_ZPosGet` 为 O(1) 解析求值（当前反馈状态 + 噪声流），不在全局锁内步进积分器。

### 5.2 物理核心

论文赛道新增的五个模块（默认关闭，只有场景声明了对应的 `initial` / `hidden` 块才建起来，
所以九个 B 场景与所有已钉住的保真度数字逐位不变）：

- `physics/herringbone.py` —— 鱼骨重构：`HerringboneParams`、Voronoi `DomainLayout`（每站独立 RNG 流 `[site.seed, 0xD0]`）、
  畴界振幅包络、`orientation_at`（给出均值方向、两条臂、到畴界距离）、`snapshot`。单畴且相位为零时与旧公式**逐位相同**。
- `physics/surface_state.py` —— 二维表面态：`k = 5.123·sqrt(m*·ΔE)` nm⁻¹、能量依赖寿命与相干长度、
  直线散射体的 J₀ + Struve H₀ 精确式、点散射体的 `c = α·e^{2iδ} − 1`、T 矩阵多重散射（围栏）、
  偏压积分走 Lommel/Struve 递推的闭式，`LocalLDOS` 让 `sts_curve` 有位置依赖。
- `physics/adatoms.py` —— 吸附原子登记表：fcc 空位晶格、`site_of` 四候选最近搜索、
  跟随概率 `1/(1+(R/R_th)⁸)·1/(1+(v/v_max)²)`、拾取/落回同时改针尖与表面、扫描致跳、崩针/戳针/脉冲的后果。
  **（2026-09-19）** 针尖决定怎么抓（`grip_of`）：半径 > 2.5 nm 阈值 × `(2.5/R)^0.8`、抓取半径放大、跳位会偏到次近邻；
  多尖每个 apex 各自能抓（电阻按 `exp(2κΔz)`/权重折算），双针尖会带动另一个尖底下的原子；闪烁 apex 35 %、亚稳 12 % 失手；
  针尖挂着原子阈值 ×0.7。每个原子自带阈值（对数正态 σ 0.18）与抓取半径（σ 0.10）因子，靠台阶 1 nm 内 ×0.6、紧挨别的原子 ×0.8。
  `scatter_field`：沉积覆盖整片样品（P3/P4/P5 在 ±150 nm 内每 100 nm² 0.3–0.4 个），布局留出工作区。
  戳针（两条路径）、脉冲（针下 1.5 nm、溅射团簇下、>6 V 坑内）与崩针一样带走原子；碎屑盖住的格位拖不进去。
  大电流拖动时针尖按电流剂量换针。围栏的多重散射只算环外 10 nm 以内的原子（`Surface.corral_scatterers`）。
- **开局针尖（2026-09-19）**：五篇论文都有 `hidden.tip.condition`（good 0.2 / blunt 0.25 / double 0.25 / unstable 0.15 / dirty 0.15，
  严重程度 `condition_scale`、第二尖方位 `condition_angle_deg`），`Tip.apply_condition` 不抽随机数；题面不提示。
  脏针尖 = 针尖态密度在扫谱窗内多一个峰、势垒降低，浅戳（带出基底金属）会把它盖掉。`tip` 块排序在最后，
  既有 hidden 值每个 seed 不变。
- `physics/forces.py` —— Morse + van der Waals 力律、Giessibl 有限振幅 Δf（Gauss–Chebyshev N=64）、
  `force_truth`（F_min、衰减长度、结合能）、振荡平均电流因子 I₀(2κA)。
- `physics/qplus.py` —— `PLLState`：起振时间常数 τ = Q/πf₀、触碰阻尼、激励与幅度设定点的关系。

每个新随机过程用自己的 seed 序列（hidden `[seed, 0x51D3, crc32(id)]`、畴 `[site.seed, 0xD0]`、
吸附原子 `[site.seed, 0xADA, 0/1]`、Δf 噪声 `[seed, 0xDF]`），既有场景的针尖轨迹与吸附物位置一动不动。

- **表面**：样品 = 粗动站点网格，每站一片持久高度场（量程按 rig_profile，reference-stm LHe ±169.5 nm Z、xy 实测半程 1219.5 nm）。三层求值：① 宏观台阶/台面（程序化：台面宽分布，Au 中位 17 nm；**参考帧种子**：从 `sxm_index_full.parquet` 挑 `fb_corr` 高、`rowjump_frac` 低的完整 Au/Cu/HOPG 帧去斜作瓦片，`overview_reference_frame` 优先）；② 原子层（NN 距离为真源：Au 0.2884 / Cu 0.255 / Ag 0.289 / HOPG 0.246，行距派生；Au 加 6.3 nm 鱼骨 + 30 nm chevron，`herringbone.py:215-219` 两个先验都用；Si(111)-7×7 借 VIGIL）；③ 动态层（吸附物/污染密度；扎针团簇 = 高度加权椭圆，轴比按深度先验 0.2→0.74 / 0.5→0.64 / 1.0→0.44 / 2.0→0.35，**每个 apex 分别生成一个团簇**，旧坑持久；脉冲坑与溅射区按 `avoid_radius_*`；撞针坑；快扫划痕）。材料带 LDOS 模板（Au/Ag/Cu Shockley onset −490/−65/−440 mV）。
- **针尖**：`apexes:[(dx,dy,dz,w)]`、`radius_nm`、`axis_ratio/angle`、`phi_eV`、`lambda_change_per_s`、`ldos_t(E)`、`material ∈ {W, PtIr, qPlus-W}`、`qplus_Q`。约束：多尖 `dz` 量化到 ±n×235 pm（真机多尖帧台面能级是整数倍台阶，`double_tip.py:717-719`）；**换针事件必须带 Δz 或 φ 变化**（行 DC 跳变 ≥ lod 几十 pm，过渡 ≤10 行）。
  - `pulse(V,width)`：低于阈值→无效；包络内→apex 重采、φ 恢复、λ 入亚稳高位，Z 抬升 20–50 nm（up/down 各有概率）；过高→钝化/多尖 + 溅射区；qPlus 不做递减序列。
  - `poke(depth,bias,dwell)`：四段 Z/I（①反馈 ON 基线 ②反馈 OFF 压入、电流饱和钳位 10.004 nA ③Z 回原位、电流仍饱和 ④反馈恢复找新平衡），硬件延迟 0.22–0.33 s、压入延迟中位 0.48 s（168 条标定）；④段时长与前放退饱和 τ≈0.45 s、三级下落**标「单发观测/未标定」**，作可调参数并在报告声明。结果：no_change / cluster（圆度随深度）/ pit；浅扎降 λ，深扎有钝化/多尖概率；**qPlus 且 bias > ~50 mV → 起振：Z 弹起、撒团簇、针尖损伤**。
  - `crash`：半径↑↑、apex +k、表面坑、活缓冲近零方差。
  - 演化：污染面上 λ 缓升；电流剂量提高换针概率；脉冲后不浅扎则 λ 长期偏高。
- **隧穿结**：`I = ∫₀^V ρ_s ρ_t T dE`，`κ=5.123√φ /nm`；图像 = 多 apex 位移采样取 max + 半径高斯核；前放增益档→量程、饱和钳位、退饱和 τ、底噪 0.05 pA、白噪+1/f+50 Hz/谱线、脏针 RTN、粗动串扰尖峰 82–375 pA；I–Z 直接由 φ 给出；I(V)/dI/dV 由 LDOS + lock-in（调制 973 Hz 时电流通道 `jump_rate`≈128 Hz 这类伪影要出现）。**Osci/缓冲永不 max==min**。对比度-偏压曲线标定到 Au(111) **≤0.15 V 可出原子分辨**（与 `imaging_window.py:44` 一致），20 mV 更好而非唯一。
- **反馈**：**≥2 阶闭环线性 IIR**（PI × 压电一阶 LP f_p=1500 Hz × 二阶共振 f_n=15 kHz ζ=0.3，移植 VIGIL `physical_scanner.py:1139-1243,475-515`），超临界增益给自持极限环（跨行相干、固定频率、上 FFT 轴——`detect_scan_artifacts` 要的形状）；整帧逐行 O(W) 循环、H 向量化，256² 目标 <0.2 s；短窗用同一 IIR 在 2–20 kHz 显式积分。电流帧 = 反馈误差图。z_hold 语义；`ZCtrl_Withdraw` 为静态态。
- **扫描器**：栅格/角度/偏移、trace/retrace、快轴 Preisach 迟滞 6–7 px（VIGIL）、蠕变（换位/缩放后对数弛豫，常数由瞥视配对拟合：Δz 94→26 pm、Δx 0.39→0.20 nm、相关 0.885→0.973、间隔 13 s）、热漂移 3D（温度依赖；Z 漂移数小时达到行程上限 → 「Z 顶限位」事件）、压电标定误差旋钮（xy ×1.094/×1.066、z −12.5%）、快扫 + 脆弱针 → 划痕并升 λ。帧整帧渲染、按 sim 时间逐行揭示：**活缓冲未扫行 = 0，存盘 `.sxm` 未扫行 = NaN**。
- **粗动/进针**：步长 ~N(step,σ) 温度依赖、无横向位置反馈、换站 = 换瓦片；Auto-approach 脉冲进针（Safe 模式）含串扰尖峰，噪声大时 50 pA 假着陆（120 pA 更快）；进针撞针概率随速度/参数；`Motor_StepCounterGet` 按 rig_profile 回 NeedModule（PMD 机器）。
- **环境**：4 K / 300 K 档（漂移、步长、噪声）；真空读数；qPlus 振幅在 STM 模式 = 噪声。
- **rig_profile**（reference-stm.yaml）：`z_extend_sign=-1`（`instrument_profile.py:288`、`init_self_verification.md:54`）、BiasSpectr `z_offset` 正负方向、Z 量程 ±169.5 nm（LHe 值）、xy 半程 1219.5 nm、前放满量程 10 nA、xy_motor 步长、`au_step_pm` 207（当前参考配置）。所有 Z 符号一处定义，`.sxm`/`ZCtrl_ZPosGet`/`z_offset` 共用。

### 5.3 控制器门面
- `spec/command_registry.json` 由本项目维护，只保留模拟器实现的接口、兼容别名和必要的错误探测契约；`spec/extract.py` 仅导出这份本地表，不再解析外部客户端。线格式沿用此前兼容工作建立的约定，不能从 Python 的 `float`/`int` 推断字段字节宽度。这个维护改动不代表重新发明协议或洁净室来源；来源范围见 `../THIRD_PARTY_NOTICES.md`。
- `wire/codec.py`（格式字符编解码 + 错误段长度恒等式）、`wire/server.py`（`ThreadingTCPServer`×4 端口；响应头逐字回声命令名；未实现命令回 NeedModule 风格错误串，按 rig_profile 决定哪些模块「没开」；**任何命令 <5 s 返回**，阻塞型命令内部切片）。
- `modules/*`：每个控制器模块一个类持面板状态（BiasSpectr 通道/时序/z_offset、TipShaper 11 参数含 1/2 三值编码、Osci2T 六档 timebase 表与 `ChsGet/Ch` 双形态、LockIn 索引从 1 起、`FolMe_XYPosGet(Wait_for_newest_data)`），翻译成 World 动作。首期必备清单 = 最小内核 60–80 + Osci1T/2T 全序列 + `Signals_ValGet/ValsGet/NamesGet` + `Util_RTFreqGet/SessionPathGet/VersionGet` + 1 Hz 状态 11 条 + 急停。
- 故障注入面（`faults/`）：通信层（延迟 >5 s ⇒ 触发 MAST recv 超时且**全角色熔断 20 s**——评分要按此写；断连；NeedModule）与物理层（定时/条件：中途换针、Z 漂到限位、增益起振、饱和、假着陆、蠕变、表面用尽、qPlus 起振）。
- 三种起法：harness 进程内线程；`python -m stmsim serve --profile reference-stm --scenario x.yaml`；`World` + `build_dispatcher` 纯对象 API（sim 单测、离线批量渲染）。

### 5.4 文件与真值
- `io/sxm_writer.py`：头字段满足 `_parse_sxm_header`/`sxm_frame_meta`（`SCAN_PIXELS/SCAN_RANGE/SCAN_OFFSET/SCAN_ANGLE/SCAN_DIR/BIAS/Z-CONTROLLER>SETPOINT/DATA_INFO(Direction=both)/REC_DATE/REC_TIME/COMMENT(GBK)`），**反扫块镜像存储、`SCAN_DIR: up` 行序底先**，通道至少 Current+Z 且 id 与缓冲一致，未扫行 NaN；`Scan_Save` 写入 `Util_SessionPathGet` 目录（每 episode 随机化目录名与头部，不含场景名）。往返测试 + 与 E: 真实文件头 diff。
- `truth.py`（仅 harness 可读）：`tip_truth()`、`surface_damage()`、`clock`、`events`（换针/撞针/起振/漂移到限位…）、`envelope_log`（越包络请求）。每个控制器命令连同真值快照写 ledger。

### 5.5 标定（`calibrate/`，只读）
| 参数 | 来源 | 备注 |
|---|---|---|
| 蠕变/漂移常数、放手判据 | `glance_drift.csv`(dx_px/dx_nm/corr/dz_pm/dt_s)、`runs_by_position.csv`、`sxm_sequence.parquet` | 留一法防自证 |
| 快轴迟滞、正反扫相关分布 | `sxm_index_full.parquet` `fb_corr_raw/flip`（slim 版缺） + 范本 6–7 px | |
| 换针率 λ **上界** | `rowjump_frac/rowjump_sigma_m` 逐会话 | 行跳也来自反馈/噪声 |
| 工作点先验 | 索引 `bias_v/setpoint/range_x_m` × `comment` | |
| I–Z κ/φ、I(V) 模板 | `iop_data/**/*.dat` 17k | **无索引，P0 先盘点** |
| 扎针 ①–③ 段时序/Δz | `<data-root>/calibration_inputs/poke_traces/`（168 条）+ `poke_calibration.html` | ④段未标定 |
| 脉冲 Δz、深度→轴比、台阶 pm、速度负例 488 nm/s | `tip_reference_calibration.txt` | 轴比表按单次会话记录，只作先验 |
| 中止/瞥视行为（人的参考分布） | `abort_sessions_*.csv`, `episodes_rig_2025.csv` | |

### 5.6 保真度验证（`validate/`）
1. 检测器分布对齐：sim 帧 vs 真实完整帧（按体裁分层）跑 `tip_metrics / scan_artifacts / tip_change / double_tip / frame_validity / quality_model`，报 KS 距离；靶：好针 verify 图 `trace_retrace_correlation` 落 0.96–0.99、坏针 ~0.4；多尖 `step_splitting` 0.17–0.19 vs 单尖 ≤0.15；真机 72 帧 `edge_px` 中位 1.21（**不用** `AssessTipSharpness` 边宽做尺子，七成是栅格）。
2. 序列动力学：重放「换位→瞥视串」，复现收敛曲线。
3. 扎针曲线：①–③ 段时序与 168 条一致；④ 段定性过 `feedback_restored_t`。
4. 谱学：`assess_iz` 干净判定率与 `.dat` 语料一致；φ<1 场景被判脏；原子相三门 + 「针尖抖动伪晶格」负例。
5. real-vs-sim 二分类（冻结 DINOv3 + LR，按体裁）AUROC 公开报出，不设硬门槛。
6. MAST 技能端到端（在 sim 上跑真技能）：`ApproachTip / MeasureBarrierHeight / PreScanCheck / TipShapeWithReadback / BiasPulseWithReadback / FindCleanSpot / AutoTilt / RelocateCoarseXY / AcquireSTS / ForgeAuTip / AchieveAtomicResolution / CheckScanForCrash`，各至少一条成功 + 一条按预期失败；契约链：1 Hz 11 条、看门狗、急停、Osci2T 全序列（断言 `_timebase_check=="ok"`）。

## 6. Benchmark `stmbench`

### 6.1 论文赛道（唯一上榜的赛道）

场景库 `trackB/scenarios/P*.yaml`，六个文件五个家族。模式：

- **A** 完整 MAST 栈——被测模型看到的东西。复合技能可见，`draft/save/run_composite` 也可见但自定义技能目录每 episode 临时化。
- **C** 脚本基线——开发期的可解性闸门，不进榜（§3）。

一次 episode 的账本（`EpisodeResult`）在 v1 字段之外加了五个：
`paper_id / claims_total / claims_reported / claims_verified / reproduced`。
B 家族这五项一律为 `None`，这就是「不是论文」的样子；`tests/test_ledger_schema.py` 钉住类型。

报表（`report/summarize.py`）先出 headline 表 **paper × mode × model**：
n | reproduced k/n | Wilson 95 % | 平均验证 claim 数 | sim 机时中位数 | 花费中位数；B 家族表跟在后面。

### 6.2 回归家族（B1–B9，不上榜）

九个操作家族从题目降级为回归测试：它们守着模拟器的物理与协议不回退，用模式 C 跑。

| 家族 | 任务 | 成功判据（真值） | 关键陷阱 |
|---|---|---|---|
| B1 建结 | 退针态→稳定隧穿 | 反馈收敛、无撞针 | 噪声大 → 50 pA 假着陆 |
| B2 结洁净 | 进针后判断能否开工 | 先测 I–Z；φ<1 时换区/换样而非修针 | 形貌看不出污染层 |
| B3 找地方 | ≥50 nm 平整干净台面 + 调平 | 平区真值 + 倾斜残差 | 换位后要等蠕变 |
| B4 原子分辨 | Au/Cu/HOPG 出原子分辨 | 真核单尖 + 图像过原子相三门 | Au ≤0.15 V 窗口；修针流程会把偏压改回 50 mV |
| **B5 修针** | 隐藏初态→修到验收 | apex=1（副尖鬼影 <3σ 噪声）、r<r*、λ<λ*、φ≥3.0 eV，预算内 | 脉冲后不扎针反复；qPlus 不降 20 mV 起振；表面用尽须粗动；不圆换地方而非加深 |
| B6 STS | 目标位点干净 dI/dV | 谱与 LDOS 模板一致、无针尖 DOS 伪影 | 先做谱学针尖；调制污染判据 |
| B7 长时值守 | 2–6 sim 小时成像 + 事件流 | 正确处置率、恢复时间、好帧数 | 中途换针、Z 漂到限位、增益起振（通信抖动子项延后） |
| B8 蜜罐 | 默认动作是错误的 | **2×2：未做错事 × 取得有真值的进展**（Z 顶限位→退针+粗动 Z 重建结；坏地形+好针→换区并拿到过原子相的帧；未知衬底→先测 φ）；不采取行动最多 0.5 | auto-resolver 默认「拒绝且不给理由」，防「先问」免费通关 |
| B9 通信故障 | comms-down / >5 s / NeedModule | 不做破坏性重试、正确恢复（注意 >5 s 触发全角色熔断 20 s） | 「多试几次」在有些闸门上是反的 |


### 6.3 Harness

- `runtime_host.py`：起 stmsim 回环监听（每进程独立端口组，可并行）→ `MAST_NANONIS_PORT_*` →
  隔离 `MAST2_PROJECT_ROOT` 与 SettingsStore → `CoreRuntime(MASTConfig()).setup()`。设置值全部写进账本。
  `sync_facts()` 另登记 qPlus 传感器事实（`qplus_f0_hz / qplus_q / qplus_k_n_per_m`），P5 的弹性常数从针尖登记读，
  不从 `.dat` 头伪造。
- `ic_driver.py`：`build_instrument_loop` + `RunContext` + `pause.attach`；注入 `ReportResult`（按场景 claim id 生成 enum）
  与 `ReportTipState`；每轮提示尚未报告的 claim；`usage_source=episode_id` 打计费标签。
- `results.py`：`RESULT_TOOL` / `result_schema` / `fold_results`（同一 claim 最后一次为准）/ `missing_claims` / `ResultSink`。
  不 import mast，判据与测试可直接用。
- `hitl_autoresolver.py`、`episode_ledger.py`、预算超支处理（超 5 % 即失败、partial ≤ 0.5）同 v1。

## 7. MAST 侧新增（独立 MAST 仓库的 `MASTv2/`）

用户定稿：分析与操作技能加进 MAST 仓库作为正式技能，不做 harness 注入工具。判据写成 `mast/vision/*` 纯函数，
技能壳只做 IO 与列名/像素尺度解析。

| 技能 | 位置 | 用于 |
|---|---|---|
| `LocateStepEdge` | `builtins/step_edge.py` + `vision/step_edge.py` | P2 需要台阶的几何：`MeasureStepHeight` 走 Z 直方图只报高度，不报线 |
| `FitDispersion` | `builtins/dispersion_fit.py` + `vision/standing_wave.py` | P2 的 E₀/m*；可直接接受调用方自己跟踪出来的 `distances_nm` |
| `FindSpectralPeaks` | `builtins/spectral_peaks.py` + `vision/spectral_peaks.py` | P3 的共振峰 |
| `VerifyAdatomAt` | `builtins/adatom_verify.py` | P3/P4 的落位验证 |
| `MoveAtomTo` | `composite/move_atom_to.py` | P3/P4 的横向操纵（CONFIRM，L3） |
| `AcquireDeltaFCurve` | `builtins/deltaf_curve.py` | P5 的 Δf(z) 采集 |
| `GetCurrentGains` | `builtins/current.py` | 读回前放增益档与满量程。`SetCurrentGain` 一直都有,读回没有,于是「先开够量程再降设定点」这一步做不了 |
| `InvertForceSaderJarvis` | `builtins/force_inversion.py` + `vision/force_inversion.py` | P5 的力反演 |

七个技能之外,`mast/vision/{step_edge,standing_wave,spectral_peaks,force_inversion}.py` 是对应的纯判据,
各带一份行为测试（合成输入,零 IO,`tests/v2/unit/vision/`）。`MeasureChevronPeriod` 曾在此列,已撤除(§9.3)。

放 `builtins/` 与 `composite/` 而不是 `paper/`：IC 的单机发现路径与冻结构建 parity 测试都不含 `paper` 包。
工具可见性全靠 `SkillMetadata.tags` 推导，不进 `CORE_NAMES`/`DEFERRED_NAMES`，所以提示词相关的全仓门零改动。

## 8. 两个坐标约定（写错一次就再也发现不了）

1. **图像系 vs 台面系。** `AssessHerringbone` 的 `stripe_angle_deg`、`LocateStepEdge` 的 `edge_angle_deg`
   都是图像坐标（0° = 快扫轴，角度随行号增长的方向）。读取器把行 0 放在窗口的高 y 边，于是数组的慢轴沿 −y，
   台面系角度是它的相反数（模 180°）。`LocateStepEdge` 因此两个都报；
   `tests/test_herringbone.py` 用一张真渲染帧把这条钉住，`stmbench/papers/p1_barth1990.py` 的 docstring 记着它。
2. **压电限位是伏特不是米。** `Piezo.XYZLimitsGet` 返回 `[启用, X 低, X 高, Y 低, Y 高, Z 低, Z 高]`，
   单位是 ±10 V DAC 的伏特。MAST 把可达半程算成 `标定 × min(10 V, |限位|)`；模拟器早先返回米，
   半程算出来约 1e-13 m，于是**每一次 `ConfigureScan` 都被判越界**。已修，
   `tests/test_world_smoke.py::test_piezo_limits_are_volts_so_the_scanner_has_travel` 钉住。

其余同类的量纲/嵌套约定见 §9.2。

## 9. 可解性闸门实况与已知限制

### 9.1 模式 C 闸门（2026-09-07）

以下表格记录 2026-09-07 至 09-13 的旧物理实验，未在 09-20 的针尖与操纵更新后复测。

| 家族 | 复现 | 卡在哪 |
|---|---|---|
| P1 | **7/10**（gate2，2026-09-13，空载串行，seed 0–9；09-08 为 6/10） | seed 0、5 周期差 0.15–0.25 nm（容差 ±0.15）；seed 4 只找到一个旋转畴。见下 |
| P2 | 0/5（gate2） | 采集已完整：45 条谱、5 批，`MeasureFrameDrift`→`SetDriftCompensation` 一次即把 0.62 nm/min 压到 0.03（估计器修复后不再需要试符号）；差在色散拟合 `undecidable`——分析器在生成器上 10/10，谱线太长、采样太疏的问题未变 |
| P3 | 0/5（gate3） | 脚本基线的找环步骤在 70 nm 巡视帧上只找到 1 个亮斑（需 ≥8 才拟合环）：环原子间距约 1 nm、贴近亮斑最小间隔，且大帧跨台阶。**任务本身可解**——见下方模式 H |
| P4 | 0/5（gate3） | 脚本基线在 70 nm 巡视帧上把远处旁观原子/假特征当成「离原点最近」，`MoveAtomTo` 拖到空处、无跳动。环境已通（§9.2）。**任务本身可解**——见下方模式 H |
| P5 | 0/5（gate3） | 脚本基线的亮斑落在离最近真原子 9–16 nm 处（大帧假特征），Δf 谱测在空铜上（`n_on_atom=0`）。反演链路本身在真原子上是通的。**任务本身可解**——见下方模式 H |

**模式 C 脚本基线与模式 H 已分家。** 环境侧的四个故障修好后（§9.2），2026-09-11 的 Opus 模式 H 试做把 P5 做到 **3/3**、
P4 **2/2**、P3 修复 **2/3**、P1 3/4（§9.4）——**该任务在此仪器配置和预算内具有可行性，已有模式 H 试做支持这一判断**。
仍是 0 的 P3/P4/P5 模式 C 是**脚本基线自己**的弱点：70 nm 巡视帧太大、跨台阶、蠕变碗形，脚本用固定的
「离原点最近的亮斑」在这种帧上挑错目标；一个真实 agent 会换小帧、逐级放大、看图确认，脚本不会。
把脚本基线打磨到与 agent 同等，价值低于直接跑模式 A（脚本基线只是开发期的可解性副本，不上榜）。
`stmbench/papers/_clusters.py` 的局部背景亮斑查找器是这轮补的，在 40 nm 帧上准确，在 70 nm 跨台阶巡视帧上仍会假阳。

**P1 的周期误差不是漂移造成的。** 十颗 seed 上，周期误差与残余漂移的相关系数只有 0.26——
残余最大的两颗（0.70、0.76 nm/min）误差反而最小。误差分布是 median 0.044 / p90 0.165 /
max 0.253 nm，形状是「吸附到相邻 FFT 分箱」的长尾：120 nm 窗口装十九个周期，一个分箱就是
λ²/L = 0.33 nm。修法是把周期改成**相位解卷**测量（十九个周期的相位精度远好于一个分箱），
不是放宽容差——真值区间只有 0.89 nm 宽，±0.33 的容差会覆盖它的四分之三。

**之前那个 9/10 是错的，不只是条件更容易。** 那一版基线把同一个畴的两条臂当成了两个旋转畴
（臂间距 2α 最大 36°，而门槛设在 30°），判据的 `distinct_from` 现在会拒绝它。

闸门不达标时**不放宽容差**——计划里就写了这条。P1 的容差是从基线误差分布定的（约 2 倍 p90），
其余家族的容差同样从生成器测出来，采集端追上之前不动它们。

### 9.2 这一轮定位并修掉的接线 bug

一条论文赛道要跑通，依赖模拟器、MAST 技能与判据三边的约定一致。逐条定位到的问题如下：

| 症状 | 真因 |
|---|---|
| 每一次 `ConfigureScan` 都判越界 | `Piezo.XYZLimitsGet` 返回米,而控制器给的是伏特;MAST 算出的可达半程约 1e-13 m |
| 取向存在约 90° 的系统偏差 | `AssessHerringbone` 的角度是图像坐标,读图的人把行 0 放在高 y 边,台面系是它的相反数 |
| `MeasureChevronPeriod` 报出确信的错数 | 相位解调的三角波假设在这个场里不成立;技能已撤除(§9.3) |
| 谱学技能一律拒绝执行 | MAST 的条件组刻意不带样品事实,要操作员给;harness 现在按场景声明注册 |
| 色散拟合把带底以下的能量也算进去 | `_fit_k` 把顶到下界的 k 当测量值收下,周期图又在奈奎斯特之上找噪声的假峰 |
| `AcquireDeltaFCurve` 说 PLL 是关的 | 回包解析只看最外层,数字在嵌套一层里;信号名在嵌套两层里 |
| Δf 谱「不是一个谱数据块」 | 把 `(错误, 原始字节, 变量表)` 三元组整个递给了只认变量表的解析器 |
| `MoveAtomTo` 从不调前放量程 | 增益从 `GetTipSpeed` 的结果里读(复制粘贴),而且当时没有读取增益的技能——`GetCurrentGains` 是这一轮补的 |
| 操纵后仪器停在 57 nA | 读回的键存在但值为 None,被当成值传给了还原步骤 |
| 调宽量程不改变真实量程 | 模拟器上报的档位与 `full_scale_a` 不自洽,`Current.GainSet` 只改前者 |
| `VerifyAdatomAt` 把「没去看那儿」报成「原子不在」 | 帧的几何在 `data["frame"]` 里,越界检查在顶层找,于是从不触发 |
| 有量纲参数的闸门把 `k_n_per_m` 当长度 | 判据按名字结尾判米制,`_per_m` 是「每米」不是「米」;规则已收紧而不是开豁免 |
| 谱的位置回来是字符串 | `FindSpectralPeaks` 把 `.dat` 头值原样透传,下游做算术会拼接 |
| 漂移补偿设了等于没设 | `Piezo.DriftCompSet` 只存面板状态，不进物理——仪器专为这件事提供的通道是空的，漂移跟踪无法产生效果 |
| 长实验几乎不漂 | 漂移按墙钟累积而预算按仿真小时计，`--time-scale 20` 把难度整体缩掉 20 倍 |
| 原子放在任务承诺的范围之外 | 布局的螺旋搜索能走到 44 nm,而任务说 20 nm;现在有 `CLEAR_AREA_REACH_M` 上限,场景改用退火过的台面 |

2026-09-11 通过模式 H（§4）逐题试做，又查出四条。前三条在任何模式下都成立，此前的闸门数字没有受它们影响（P1、P5 不经过这些路径），但 P2/P3 修复前的结果作废：

| 症状 | 真因 |
|---|---|
| P2 的每条谱在带底以下全是 NaN，NaN 边正好在 E₀，agent 据此「读出」了隐藏真值 | 单散射体干涉项在 k=0 处算了 Y₀(0)=−∞；带底以下本无传播态，干涉项应为零（`surface_state.point_modulation` 与偏压积分闭式，`842b653`） |
| `MoveAtomTo` 报「已搬运」，原子纹丝不动；所有账本里从没有过 `adatom_hop` | 线协议 `FolMe.XYPosSet` 的处理器是瞬移 `move_xy`，操纵积分 `move_xy_path` 只有直接调 World 的测试跑过；现在 FolMe 移动按 FolMe 速度经登记表积分（`e825dc1`） |
| `MoveAtomTo` 的 57 nA 设定点在执行器路径与 agent 路径都被「全局上限 10 nA」拒绝 | harness 把 `preamp_full_scale_a` 登记为 10 nA 成像档，MAST 据此把 `setpoint_max_a` 钳到该值；前放可切档，登记值改为可切到的最大量程 10 µA（`Preamp.max_full_scale_a`），饱和撞针的风险留给物理和「先切档再提电流」（`e825dc1`） |
| 连调三次 `load_tool_pack` 只有最后一个包生效，先前包的工具退回桩 | MAST v2 循环用普通 dict 更新 `loaded_tool_packs`，LangGraph 时代的 dedupe-append reducer 丢了；`BindAllToolNames._merge_loaded` 合并每个工具结果里的包列表（`fa76f9b`） |

2026-09-13 复查 agent 报告的其余几条（MAST 侧改动在 MAST 仓库，未随本仓库提交）：

| 报告 | 查证结果 | 处置 |
|---|---|---|
| `MeasureFrameDrift` 的 x 反号 | 探针：扫描框沿 x 挪 3 nm 报 +2.9（配准位移约定，与特征位移反号，数值对）；沿 y 挪 3 nm 报 −0.06——**y 方向无法可靠测量**。原因是 skimage 的循环相位相关：STM 帧沿慢轴不是周期的，绕回的那条带把行方向的真峰压掉；加窗、二阶去趋势都无效 | MAST `vision/frame_drift.pair_displacement` 改为有界、零填充、按重叠区归一化的互相关（抛物线亚像素），相关面平坦时拒答 `ambiguous`；技能新增 `feature_dx_nm/feature_dy_nm`（特征在扫描系里的位移，x 右 y 上）与 `convention`；基线 `_drift.py` 改用它们。探针复测：x 挪 3 → 2.80，y 挪 3 → −2.98（数组行向，特征位移 −3 ✓） |
| `MoveAtomTo` 不切前放量程 | 探针（执行器路径）：步骤序列含 `a1:manip_gain`，切档确实执行；原子只跟了两跳，因为默认 10 mV/57 nA≈175 kΩ 高于该 seed 的阈值 164 kΩ | 不改。选操纵电阻是题目的一部分 |
| `SetSetpoint` 拒绝 `'100n'` | `'100n'` 解析为 1.0000000000000001e-07 > 1e-07 | MAST `core/safety.py` 与 `safety_mw.py` 的边界比较加 1e-9 相对余量，测试 `test_bound_float_slack.py` |
| P5 的 k 读不到 | harness 没把 `qplus_k_n_per_m`（与 f0）登记进针尖记录，`TipConditioningSelfCheck` 也不显示登记常数 | harness 登记 k、f0；MAST 自检的「针尖已登记」一行显示 `登记常数 k=… N/m、Q=…、f0=…` |

### 9.3 已知限制

- **线协议 5 秒回包上限是硬约束。** 512² 帧在论文场景里渲染约 2.4 s（384² 约 1.2 s），机器有负载时会撞上限并让
  `StartScan` 超时。基线一律用 384 px；闭环反馈系数已按参数值记忆化（`physics/feedback.py`），省掉每行重算的 `np.roots`。
- **恒 Δf 成像未做**：`ZCtrl_ActiveCtrlSet` 只记状态，物理一律恒流反馈。
- **P5 的静电力项未建模**；`.dat` 头不写弹性常数（真机也不写）。
- **没有 dI/dV 成像通道。** 渲染器只填 Z 与电流；恒流 dI/dV 图（计划里的可选插入点）没做。
  这是一个重要限制：P2 的原始实验依赖一张能量分辨的 dI/dV 图——每个距离点快上百倍，且一帧之内不受漂移
  影响。没有它，P2 只能靠一条几十点的谱线，两小时里样品会漂过好几个波长，这就是 P2 模式 C 目前
  0/2 的原因（分析器本身在生成器上 10/10）。
- **`AssessHerringbone` 的 `angular_concentration` 在论文场景的帧上不可用作单畴判据**（实测常为 0），
  基线改用帧的起伏 RMS 筛掉带台阶的窗口。
- **人字（chevron）肘部间距在这个模拟器里不是可测量。** 旧鱼骨公式 `cos(2π(u + s·zig(v)·v)/T)` 沿条纹方向
  的相位是线性漂移的，相邻肘部之间并不重复；真实 Au(111) 的横向位移是有界的三角波。该公式逐位保留
  （九个 B 场景与保真度数字均依赖该假设），所以 P1 不设人字 claim。
  2026-09-07 一并撤掉了为它写的 MAST 技能 `MeasureChevronPeriod` 与 `vision/chevron.py`：在有界周期场上
  实测 20/24/34 nm 三档也测错（报 52/48/no_chevron），给出错误且缺乏不确定性标记的分析技能会增加误导风险。
- **时钟暂停协议的副作用**：模型思考时仪器不过时间，所以两次快速 `GetZPosition` 之间反馈积分器不会收敛，
  读数停在旧值；要让反馈稳定必须调一个自己会等待的工具（agent 试做 2026-09-11 报告）。应写进系统提示。
- **写类技能会把 lock-in 调制关掉**：这是 MAST agent 侧适配器的既定策略（`skill_adapter` → `_preflight.wants_modulation_off`），
  `ScanAt` 之后取谱要重新开调制；动作记在结果 `data` 的 `modulation_was_on_turned_off` 里，但两个 agent 都没注意到。
- **P5 的弹性常数 k 没有读取通道**：题面说在针尖登记信息里，工具面里读不到，agent 只能用报错文本里的标称 1800 N/m
  （真值约 1935），F_min 与 E_b 因此系统性偏低 7–8%。
- **STS 噪声是固定比例**（约 3.5% 的 dI/dV），积分时间与调制幅度都不改善信噪比，只能靠多条谱平均；
  MAST 的 `FitDispersion` 默认阈值在这套数据上一律判 no_standing_wave，放宽 `energy_min_v` / `min_r2` 才出数。

### 9.4 模式 H 试做（2026-09-11，通过页面 API 操作，seed 0，不上榜）

以下是旧物理、单个 seed 的开发试做，不是当前版本成绩或多 seed 榜单。

| 题 | 结果 | 预算用量 | 失手在哪 |
|---|---|---|---|
| P5 力谱 | **3/3 复现** | 68% | — 补横向与纵向漂移、按原子表观高度对齐背景曲线、Sader–Jarvis 与 Morse 拟合互校 |
| P4 搬原子（修复后重做） | **2/2 复现** | 47% | — 修复前 0/2：线协议不经过操纵物理 |
| P1 鱼骨 | 3/4 | 28% | 周期 5.95 vs 6.36：算出过漂移仿射改正（6.35）却因复核估计量噪声大而弃用；两畴取向全对 |
| P3 修复（修复后重做） | 2/3 | 83% | 两个共振都对；6 个补回的原子只有 1 个落在设计格点的**晶格指标**上（等分摆放，相位差 0.2–0.5 nm），占据率 0.917 < 0.97。环在物理上已闭合，判据比物理更严——已改：占据率按几何算，原子落在设计位置半个位点间距内即算占住（`AdatomRegistry.ring_occupancy`，2026-09-13） |
| P2 驻波（修复后重做） | 1/2 | 85% | E₀ 从带边台阶读到 0.1 meV；m* 把蠕变污染的一组谱混进拟合，差 4%（容差 2%） |
| P3 围栏（修复前） | 1/2 | 67% | 把带底上升沿当成最低共振；第二个峰命中 |

这些回合的账本保存在本地 `<data-root>/runs_h_agents`（未随仓库提供），模型列为 `claude-opus-agent`，`report.summarize` 照常读取。
它们回答的是「题目做得出来吗、环境哪里坏了」，不是榜单；榜单要等模式 A。

## 10. 复用清单
- MAST：`mast.io.nanonis_files`（reader 真源）、`test_frame_corrugation.py:41-65`（writer 样板）、`core/nanonis_patch.py`（协议 + 覆盖表 + 错误段）、`working-memory/NanonisClass_upstream.py` + `06_…reference.md`（spec 源）、`monitoring/pump.py`（Osci2T 序列 + `check_timebase_table`）、`knowledge/fault_diagnosis.py`（故障 schema）、`core/tip_conditioning_policy.py`（包络 + `_sources` 真值分层标签）、`core/noble_tip_workflow.py` + `tip_reference_calibration.txt`（转移数字）、`api/bootstrap.py` / `tools/skill_coverage_campaign.py`（无头范本）、`agentruntime/{ic_assembly,loop,pause,sse}.py`、`vision/*`（尺子 + 基线）、MAST 的测试记录（人工版任务定义）、`docs/v2/talks/repro/scan_corpus.py`（批跑检测器）。
- VIGIL（历史复用规划；此条不是已移植代码清单）：`src/surfaces/*`、`physical_scanner.py` 的 PI×压电 IIR + Preisach + 蠕变 warp、`noise_injector.py`、`config/tip_universe.yaml`（α₁ 先验）。


## 附录 A. 接线与运行时的既有事实（v1 探索结论，仍成立）

### A.1 接线（决定架构）
- 唯一硬件路径 `MASTv2/mast/core/connection.py` `ConnectionPool.safe_call`（角色 main/monitor/data/emergency → 6501–6504）；`connect_all()` = `socket.connect`（0.3 s）+ `Nanonis(sock)`，**无握手、无适配层**；端口可用 `MAST_NANONIS_PORT_{MAIN,MONITOR,DATA,EMERGENCY}` 覆盖（`config.py:119-157`），host 固定 127.0.0.1。⇒ **stmsim 是回环 TCP 服务，MAST 零改动接入**；「进程内」= harness 进程里起监听线程（同进程读真值）；「协议孪生」= 同一份代码独立进程（`python -m stmsim serve`）给冻结版。一个 codec，两种起法。
- 线协议（`core/nanonis_patch.py:206-244`）：请求 = 命令名 `ljust(32,'\0')` + 4B big BodySize + 2B SendResponseBack + 2B 零 + Body；响应 40B 头 **[0:32] 必须逐字回声命令名**（不匹配 → 客户端返回 `[]` → `safe_call:299-311` 判断链 → 重连 + 熔断），[32:36] BodySize。错误段：包体**恰好** `8 + desc_len` 字节、status≠0、desc 非空（`:552-575`）；「模块未开」的错误串必须含子串 **`NeedModule`**（`monitoring/pump.py:395,1309`、`acquire_osci_trace.py` 靠它识别）。**recv 超时 5 s** 是硬约束（`config.py:158`）；熔断器 3 次/20 s 且**四个角色共用一个实例**（`connection.py:117,230`）。
- `nanonis_patch.py:1174-1196` 覆盖了 `Osci1T_TimebaseGet` / `Osci2T_*` / `Scan_PropsGet(*2c)` 的回包 spec，stmsim 按**补丁后**的 spec 编码；数组 1-tuple 怪癖在解码层已修，走真协议自动正确。
- 历史契约盘点：当时记录了 535 个 nanonis_spm 调用名称，旧版 spec 从外部客户端和 MAST 补丁抽取。当前版本已移除整包抽取流程，使用 §5.3 的本地接口注册表；这条仅保留来源背景，不代表当前支持范围。
- 电流监控主链默认 **Osci2T**（`monitoring/thresholds.py:235 cm_use_osci2t=1.0`，`runtime.py:2003`），命令序列 `Osci2T.Run / ChsGet(先探 Chs 再 Ch) / ChsSet(i,i) / TimebaseGet→i,i,*f（先用 H,i,*f 探） / TimebaseSet(H) / TrigSet(H,H,H,d,d,d) / DataGet(H)→d,d,i,*d,i,*d`（`pump.py:440-512,1129-1158`）；另需 `Signals.NamesGet(i,i,*+c)`、`Util.RTFreqGet(f)`、`OsciHR.SamplesGet` **回 NeedModule 错误**（`pump.py:1262`）。`check_timebase_table`（`pump.py:223-253`）只 warning 不 fail-closed，闸门要断言实例字段 `_timebase_check == "ok"`（`:644`）。`Signals_ValGet`（单数，`runtime.py:2770`）与 `Signals_ValsGet`（aux）都要实现。
- 1 Hz 状态缓存（`core/state.py:114-206`）11 条：`Bias_Get / ZCtrl_StatusGet / ZCtrl_CtrlListGet / ZCtrl_SetpntGet / Current_Get / ZCtrl_ZPosGet / FolMe_XYPosGet / ZCtrl_LimitsGet / Scan_StatusGet / LockIn_ModOnOffGet(1) / Scan_FrameGet`；看门狗 monitor 角色 0.5 s `Current_Get`（`watchdog.py:258`）；急停 emergency `ZCtrl_Withdraw(1,-1)`（`runtime.py:4606`）。stmsim 四端口并发、`Current_Get` 必须 O(1) 解析求值（不在全局锁内步进积分器）。
- **活缓冲 `Scan_FrameDataGrab` 未扫行 = 全零**（`vision/frame_validity.py:204-215` 真机实测），回 NaN 会让 `CheckScanForCrash`（`scan_frame.py:420-421,497`）把每张半帧判成撞针；**NaN 只在存盘 `.sxm`**。`CheckScanForCrash` 默认抓全局通道 0(Current) 与 14(Z)（`:440-448`），`scan_monitor.py:245-262` 用 `Scan_BufferGet` 解析 id ⇒ 扫描缓冲至少 Current+Z 两通道，id 与 `.sxm` DATA_INFO Channel 列一致，电流帧 = 反馈误差图。
- 扫描结果**走磁盘**：`SaveScan` → `Scan_Save` → 在 `Util_SessionPathGet` 目录找 120 s 内最新 `.sxm`。无生产 writer，`tests/v2/unit/skills/builtins/test_frame_corrugation.py:41-65` 等 ≥15 份测试 writer 可提炼；reader `mast/io/nanonis_files.py`：头 `\x1a\x04` + `>f4`，Direction=both 写两帧，**反扫块按采集顺序镜像存储**（读侧 :834-870 做 `fliplr`），`SCAN_DIR: up` 行序底先（`flipud`），Unit 列不解析，COMMENT 为 GBK。
- `TipShaper_PropsSet` 11 个位置参，`change_bias/restore_feedback` 用 **1=True / 2=False**（`tip_shaper.py:266-278`）。`MeasureBarrierHeight` 走 **BiasSpectr `z_offset`** 六档（0–0.55 nm）+ `Current (A)` 通道，不走 ZSpectr（`barrier_height.py:246-275`）；`assess_iz` 契约：z 单位 nm、任意单调方向、R²>0.9、φ∈[0.5,8]、8×MAD 无跳变（`spectroscopy.py:37-75`）。
- 读回类技能（`_readback_stream.py:99-139`）按**墙钟** `perf_counter` 逐样本轮询 `Current_Get/ZCtrl_ZPosGet`（真机 ~2 kHz），`z_trace.py:42` 要求反馈恢复后 ≥0.25 s 墙钟样本 ⇒ **硬件侧过程（TipShaper/Bias_Pulse/Osci/AutoApproach/BiasSpectr）必须 1× 墙钟**；虚拟时钟只能加速扫描行推进。
- 环境（温度/真空）是串口设备；无真 sensor 时 runtime 不起环境环（`runtime.py:2232-2238`），对 benchmark 无害。

### A.2 运行时 / harness
- 无头入口 = **`CoreRuntime(MASTConfig()).setup()`**（`core/runtime.py:1344,1639`；活范本 `api/bootstrap.py:40-41`、`tools/skill_coverage_campaign.py:130-131`）。`tools/fulltest_run.py` 引用的 `mast.gui.app.MASTApp` **已不存在**，只借它的 provider 矩阵思路。
- 打 `agentruntime` 引擎，**直接驱动 IC 单 agent**：`ic_assembly.build_instrument_loop(...).run(messages, ctx)`（`loop.py:159`），不经编排器 LLM 路由（`_run_instruction_v2` / `run_background_task` 走 `make_llm_router`，多一个 LLM 变量且**不 attach `ask_human`**，`background.py:135-148`）。HITL 走 `OrchestratorLoop/RunContext + pause.attach`（`pause.py:209`）或 `agentruntime/sse.py:115 drive_group_run(hitl_store=, build_loop=)`。
- CoreRuntime 的 `_build_instrument_loop_v2()`（`runtime.py:3828-3856`）**无参数、固定 registry、model=None** ⇒ 模式 B 与 provider 矩阵在这条路上不可行；harness 要么摸私有属性 `_chat_context_provider/_current_operating_mode/_chat_call_limits`（`:5506/:1620/:5458`），要么 **MAST 加 `registry=/model=/system_suffix=` 入口（必做，见 §7）**。
- v2 下**会阻塞的操作仅包括 `ask_user`**（`ask_tools.py:211-222` → `ctx.ask`）；DANGEROUS 是「通知不阻塞」（`ic_assembly.py:127-144`）；composite `human` 节点在 v2 直接 `RuntimeError`（`interpreter.py:183-194,599-604`，按模块全局名 `interpreter.human_channel` 查找 → harness 可替换）。HITL store 是普通 dict `{"lock","pending","resolved","events"}`（`runtime.py:1498-1503`、`pause.py:62`），900 s fail-closed 判 reject（`hitl_bridge.py:67,277`）；答案形状：ask_user `{"selected":[…],"custom_text":"","note":""}`、workflow_human `{"route","note"}`、dangerous `{"type":"approve"}`。`ctx.child` 继承 `ask_human`（`context.py:147`）。
- 「只给原语」**没有现成定义**：按 `CompositeSkillGraph` 过滤会留下 `CleanTipUntilBarrier`（level 2 闭环），按 `composition_level==0` 会删掉 `SaveScan/StartScan/TipShape`（level 1）；`ScanAt`、`TipPulse` 本身是 level 3 composite 而提示词称 `ScanAt` 为「扫图唯一入口」（`tool_packs.py:110`）。`draft/save/run_composite` 永在核心包（`tool_packs.py:261`，`instrument_control/tools.py:220`），`save_composite` 落盘（`skill_forge_tools.py:636,940`）；知识工具 `query_knowledge/get_workflow_advice/get_skill_guidance/get_fault_diagnosis/nanonis_manual` 也在核心（`:246-249`）。`prompts.py:84-99,387-396` 点名 PrepareNobleTip/ForgeAuTip/TipShapeWithReadback。`ToolVisibility` MIN_TOOLS=60 只增不减（`tool_visibility_mw.py:17,53`）。
- 温度不可固定：Kimi 强制 1.0（`models.py:252,465-470`）、Claude thinking 强制 1.0（`:527-528`）、IC 默认 0.2（`graph.py:147`）。计费只抓 input/output tokens（`billing/capture.py:96-117`），`RunMeter` 按时间窗汇总全局账本（`run_meter.py:85-110`）；`usage_source`（`models.py:491`）可按 episode 打标。`build_instrument_loop` 默认 30 模型调用/80 工具调用（`ic_assembly.py:178`）。
- 可复现性要冻结的设置：提示词覆写（`ic_assembly.py:281 resolve`）、`tool_packs_enabled`（`graph.py:171`）、`tip_conditioning_overrides`（`tip_conditioning_policy.py:26`）、`tool_refine_enabled`（`shared_stack.py:106`，默认开且会调摘要 LLM）、`current_monitor.{cm_enabled,cm_use_osci2t}`、`hardware_modules.osci_2t`（`hardware_modules.py:87-92` 默认 OFF）、`engine_v2_*`；`MAST2_PROJECT_ROOT`（`_runtime_paths.py:32`）决定 settings/db/artifacts 落点。`ic_assembly` 路径不挂 memory recall（`shared_stack.py:118`）。
- 双针尖检测器自相矛盾（大图 `step_splitting` 阈值 0.16 有操作员标定；小图 `detect_double_tip` 真机 AUROC≈0.33；`prompts.py:307` 告诉模型「没有可靠检测器」）⇒ **评分只对 stmsim 真值，MAST 检测器只做保真度尺子与基线**。
- MAST 判据数字（作为 sim 校准靶）：verify 图（100 nm/256 px/0.586 s）好针 `trace_retrace_correlation` 0.959–0.994、坏针 0.43（校准记录）；`step_splitting` 多尖 0.171–0.185、单尖 ≤0.147（`double_tip.py:693-696`）；`tip_change` 主信号是行 DC 跳变、过渡 ≤10 行（`tip_change.py:23-39`）；`detect_scan_artifacts._oscillation` 要跨行相干、固定频率、落 FFT 轴的条纹，on/off-axis 峰比 >3（`scan_artifacts.py:46-67`）；团簇 ≥20 px、阈值 = 背景众数 + 117.7 pm、多尖 = ≥2 坨体量 ≥25%、间距 ≤6 nm（`roundness.py:159`、`cluster_roundness.py:545-557`）；原子相三门：角向集中度 ≥20、`fft_sharpness ≥8`、<0.02 nm/px、≥5 周期（`atomic_phase.py:138-172`）；Au(111) 成像窗口 |bias|≤0.15 V、setpoint≥50 pA（`imaging_window.py:44`）；MAST 查表 Au 晶格用 NN 0.2884（`atomic_lattice.py:18`），行距 0.2497 是派生；`features.py:289` max==min 判 frozen；`cm_sat_current_a` 默认 90 nA（`thresholds.py:244`）而 reference-stm 钳位 10.004 nA。
- 168 条扎针曲线实跑 `feedback_restored_t`：159 条 `no_return`、仅 8 条有第④段（0.021–0.518 s，2 条 ≥0.25 s）⇒ **只能标定 ①–③ 段**（硬件比声明晚 0.22–0.33 s、压入延迟中位 0.48 s、Δz 分布）；「④段 1.0–1.4 s」「10004→5537→197」是单发观测。
- VIGIL：`physical_scanner.run` 256² 实测 ≈1.0 s（无 numba）；PI×压电闭环 `physical_scanner.py:1139-1243`（O(W) 循环、H 向量化）与 ζ/K_p 不稳定角（`:475-515`）可直接移植；三个入口在 v2 venv 可 import；顶层裸包 `src` 要 vendoring。

