# STM-Bench

[English](README.md) · **中文**

一台**软件扫描隧道显微镜**，以及围绕它提出一个问题的基准：

> 给模型一台模拟 STM，它能复现一篇经典而简单的 STM 论文吗？

## 为什么操作 STM 很难

STM 让导电针尖贴近表面扫描，并测量微弱的隧穿电流。在恒流成像中，反馈系统随扫描运动调整针尖高度。
电流与针尖—样品间距近似呈指数关系：原子尺度的灵敏度，也意味着微小扰动就能影响测量。
图像衬度和谱同时取决于样品与针尖，因此解释某个特征之前，需要判断测量条件是否可靠。
成像原理及其假设见 [Tersoff–Hamann 理论](https://doi.org/10.1103/PhysRevB.31.805)。

STM-Bench 模拟了其中几类困难，它们让操作成为持续的控制与诊断过程：

| 操作难点 | 智能体在这里需要应对什么 |
|---|---|
| **针尖状态是隐藏的。** 钝针尖或多尖端针尖可能使表面特征模糊或产生重影；针尖电子态也可能改变谱。 | 结合图像、谱和仪器信号评估测量质量，判断是否需要修整针尖，并检查效果。诊断仍可能存在不确定性。 |
| **控制参数相互影响。** 偏压和电流设定值影响隧穿结；反馈增益和扫描速度影响针尖跟随表面的能力。 | 为下一次测量选择参数，检查噪声、饱和或反馈不稳定。适合成像的设置未必适合原子操纵。 |
| **坐标随时间和历史变化。** 热漂移、压电蠕变和迟滞会使指令位置与样品之间的对应关系发生偏移或畸变。 | 在多次扫描和谱测量之间重新定位特征、跟踪漂移，并验证补偿效果。旧的目标坐标可能已经失效。这些扫描器效应见 [Yothers 等](https://arxiv.org/abs/1611.00243)。 |
| **动作会改变后续测量。** 修整针尖可能改善针尖，也可能使其恶化；原子可能只跟随移动一部分、保持不动，或扰动邻近原子。 | 干预后再次观察，并根据实际结果调整操作。模拟器中的操纵结果取决于结电阻、速度、针尖状态和局部环境；[Chen 等](https://www.nature.com/articles/s41467-022-35149-w)讨论了真实 STM 原子操纵中的相关挑战。 |

因此，每项任务都需要在仪器时间预算内完成**观察 → 诊断 → 调整 → 测量 → 验证**的循环。
目标值随 seed 变化，数值类结论既需要报告结果，也需要符合要求的采集证据。
裁判检查这些测量是否发生过，但不能证明报告值确实由这些数据计算而来。

这里采用的是针对部分 STM 难点的简化模型；物理保真度和向真实仪器的迁移需要另行验证。
更全面的实验挑战介绍见 [MAST-public 的 STM 简介](https://github.com/HamsterPark/MAST-public#what-is-stm)。

## 任务 4：移动一个原子

把一个原子移到目标位置，同时让附近原子保持原位。下图汇总了四个较早的回放回合、
五个选取的 Cursor 会话回合，以及 Jev Choice 的五次开发试跑。
这些回合均采用 P4 easy、seed 0、模式 H，以及维护者的完整 MAST 环境。
每项得分统计两项已验证检查：原子到达目标位置，且邻近原子保持原位。

![P4 easy 已验证检查：较早的 Luna 0/2、Terra 0/2、Sol 0/2、Astra 2/2；后续的 Opus 2/2、inherit 2/2、Sol medium 1/2、Grok 1/2、Composer 0/2；Jev Choice 的五次开发试跑均为 0/2。](docs/assets/p4-homepage/p4-results.svg)

这些是单个 seed 下的开发结果，不是成功率，也不是模式 A 榜单。
`inherit` 是记录中的模型标签，其底层模型未知。Jev 使用独立的 choice 驱动，各次试跑之间调整过设置。
[查看运行 ID 和评分背景](docs/assets/p4-homepage/README.md#later-five-model-comparison-and-jev-choice)。

## 回看两次较早的尝试

一次移动可能未到位、向后退，或完全不发生：模型必须再次观察并调整。
四次较早的试跑采用相同的起始场景和 MAST 支持，每个模型各一个回合。

**Astra · 看移动路径与后续修正。** 原子未能一步到位，有时后退，有一次保持不动。
Astra 重新扫描并调整操作，直到原子到达目标，且邻近原子经验证保持原位。

**▶ 观看 24 秒循环回放（自动播放）**

白色光点跟随针尖指令，青绿色轨迹跟随原子。全程保留概览画面。
每次尝试从操纵切换到 **512 倍速重扫**，随后暂停，展示下一次调整；微小修正会标为慢动作。

![全程保留概览的动态回放：明亮的针尖指令标记与记录中的原子轨迹展示了四次尝试、加速重扫、调整，以及验证到位的过程。](docs/assets/p4-homepage/astra-replay.gif)

**Terra · 误导性警告使进展停止。** 诊断工具的误报引发了三次未成功的恢复尝试；Terra 随后停止，没有宣称成功。

**▶ 观看 16 秒循环对比（自动播放）**

![动态对比：Terra 检查候选目标、收到诊断误报、尝试三次恢复，随后停止，未移动原子。](docs/assets/p4-homepage/terra-replay.gif)

路径和标签依据运行记录添加；这些是浓缩回放，单次试跑不能用于估计成功率。
[查看静态分镜和试跑详情](docs/assets/p4-homepage/README.md)。

## 关于这个基准

**Alpha 阶段的研究原型。** 本次发布包含模拟器、论文场景、核查测量证据的裁判、回合运行框架和回放工具。
模拟器可独立于 MAST 使用；完整基准回合需要单独的 MAST 运行时。
公开的 [MAST-public](https://github.com/HamsterPark/MAST-public) 仓库是精简源码版，
不是开发本基准时使用的完整 MAST 环境。目前，两个公开仓库尚不能让读者按作者的条件复现完整基准回合。
模式 A 尚未发布榜单。历史证据一节中的较早试跑发生在当前物理实现之前，尚未在本次发布版本上重跑。

任务以仪器操作为核心，以数据分析为辅助。不划分技能等级，也不设人类锚点：本基准不回答人类能否完成任务。
为使记住文献答案不足以过关，目标量会在有物理依据的范围内按 seed 抽取；
裁判同时要求采集证据，以及落在容差范围内的报告值。

- **`stmsim`** — 一台模拟 STM，在 `127.0.0.1` 上实现 SPM 控制器 TCP 线协议，
  兼容的控制客户端可以通过它操作模拟仪器。协议兼容不意味着与真实显微镜等效。
  针尖损伤（钝化、多尖端、不稳定或隧穿结污染）是核心因素；模型还包含进针、隧穿结、反馈动力学、
  带漂移/蠕变/迟滞的扫描、谱测量、通过脉冲和扎针修整针尖、粗动及通信故障，
  并提供供基准评分使用的**隐藏真值**。在此基础上，论文场景加入了带旋转畴的 Au(111) 鱼骨重构、
  Cu(111) 表面态及其驻波、支持横向操纵的吸附原子注册表、量子围栏的多重散射，
  以及带锁相环（PLL）的 qPlus 频移通道。可独立运行（`python -m stmsim serve`），也可在进程内使用。
- **`stmbench`** — 五个论文场景家族：

  | | 论文 | 需要复现的内容 |
  |---|---|---|
  | P1 | Barth et al. 1990 | 鱼骨重构条纹的周期、取向，以及两个旋转畴 |
  | P2 | Crommie / Hasegawa 1993 | Cu(111) 表面态的带底和有效质量 |
  | P3 | Crommie et al. 1993 | 量子围栏内的受限共振；第二个变体还要求先修复围栏 |
  | P4 | Eigler & Schweizer 1990 | 将任意一个孤立 Fe 原子移动 4 nm 到目标晶格位点，邻近原子保持原位 |
  | P5 | Sader–Jarvis 2004 / Huber 2019 | 从 Δf(z) 得到短程力最小值、衰减长度和结合能 |

  报表只回答一件事：**论文 × 模型的复现率**。

设计记录：[`docs/DESIGN.md`](docs/DESIGN.md)（中文）。操作指南：[`docs/USAGE.md`](docs/USAGE.md)。
代码审查与贡献指南：[`AGENTS.md`](AGENTS.md)，包含源码/测试对应关系和验证边界。

## 快速开始：无需 MAST 或 API 密钥的模拟器

在本仓库检出目录中，使用 Python **3.13**：

```bash
python -m pip install -e ".[dev,bench]"
python -m stmsim serve --profile reference-stm --seed 0 --material "Au(111)"
```

第二条命令启动本地控制器服务器，不提供图形界面。连接兼容的 TCP 客户端即可操作，按 Ctrl+C 停止服务器。
这条路径用于运行模拟器，不会运行论文回合。无需 MAST 或私有标定数据的测试可这样运行：

```bash
python -m pytest tests -q -m "not requires_mast and not requires_data"
```

## 五点架构概览

1. `stmsim.physics.World` 整合表面、针尖、隧穿结、反馈和扫描器，使用两种时钟：硬件侧过程用墙钟时间，
   扫描行和慢物理过程用可选加速的模拟时钟；`World.truth()` 提供隐藏状态。
2. `stmsim.modules.*` 将这个世界呈现为控制器模块（Bias、ZCtrl、Scan、FolMe、Motor、AutoApproach、
   TipShaper、BiasSpectr、LockIn、Signals、Osci1T/2T、Util 等）；`stmsim.wire` 在四个回环端口上提供服务，
   严格采用 [`stmsim/spec/command_registry.json`](stmsim/spec/command_registry.json) 中列出的请求/响应帧格式
   和命令字段类型。
3. `stmsim.scenario` 与 `stmsim.faults` 从 YAML 场景构造初始世界，为指定 seed 抽取隐藏值，
   并安排故障（扫描中针尖变化、Z 漂到限位、增益振荡、超过 5 秒的通信延迟、NeedModule 等）；
   `stmbench.trackB.truth_criteria` 将隐藏真值转为判定结果。论文场景由 `claims_verified` 评分：
   报告值类结论既要求数值在容差内，也要求回合事件日志中的采集证据；状态约束则直接核查隐藏真值和所需证据。
4. `stmbench.harness` 在模拟器上承载 MAST 的 `CoreRuntime`（端口、隔离的项目根目录、针尖/实验/真空注册），
   按固定协议运行脚本基线（模式 C），或使用待测模型驱动 MAST 的仪器控制智能体（模式 A / B0 / B1），
   并为每个回合写入一份账本。
5. `stmbench.report` 将账本汇总为论文 × 模式 × 模型表；B 家族作为回归行列在后面。

控制器命令名和字节格式是兼容性契约，规定客户端如何请求操作；模拟结果由本项目的处理器和物理模型计算。
重命名这些命令也需要修改客户端。在本地维护模拟器的命令定义，不会使它成为新的线协议，
也不能据此认定它兼容全部控制器命令。

## 完整基准的依赖：MAST

模拟器的核心依赖是 numpy、scipy 和 pyyaml。前面的安装命令也包含测试和报表依赖，但**不会**安装 MAST。

**MAST 提供什么。** MAST（Modular Autonomous SPM Toolkit）是独立的 STM 智能体与仪器控制系统。
回合运行框架使用它的 `CoreRuntime`、仪器技能和智能体循环；回放与模式 H 页面使用它的 `.sxm` 读取器。
它修补过的 `nanonis_spm` 客户端也是集成检查中验证线协议编解码的参照。

**公开可用范围。** [MAST-public](https://github.com/HamsterPark/MAST-public) 发布了部分 MAST 6.5.0 源码，
包括通用智能体与仪器控制组件，但[省略了部分专用模块、知识资产、模型权重、标定和现场配置](https://github.com/HamsterPark/MAST-public/blob/main/docs/OPEN_SOURCE_NOTES.md)。
它不是本基准开发试跑所用的完整 MAST 环境。仅凭 STM-Bench 和 MAST-public，读者无法按记录中的条件复现
完整的模式 A、模式 C、模式 H 基准回合或展示的试跑。目前尚无经过验证的公开端到端复现配置。
私有标定语料和历史回合账本也未发布。独立模拟器以及不依赖 MAST 的物理、场景和裁判检查仍可使用。

**针对完整 MAST 环境。** 下面的命令描述维护者的依赖路径，不是将 MAST-public 变成完整环境的安装说明。
使用 MAST 的虚拟环境（其中含有它自己的依赖），并让其 `MASTv2` 包目录可被导入。
如果检出目录尚未加入导入路径，可使用：

```bash
export MAST_ROOT=/path/to/MAST
export PYTHONPATH="$MAST_ROOT/MASTv2${PYTHONPATH:+:$PYTHONPATH}"
```

仅设置 `MAST_ROOT` 时，只有测试会自动把该目录加入导入路径；命令行工具和回合运行框架要求 MAST 在启动前已可导入。
模式 A / B0 / B1 / C / H 需要完整且兼容的 MAST 环境。`requires_mast` 标记检查的是能否导入；
仅通过这项检查，不能证明回合兼容性，也不构成基准结果的复现。

## 运行测试

```bash
python -m pytest tests -q -m "not requires_mast and not requires_data"  # public checkout
python -m pytest tests -q -m "not requires_mast"       # what CI runs (no MAST installed)
python -m pytest tests/test_no_machine_paths.py -q     # public-release path guard
```

测试标记包括 `requires_mast`（MAST 可导入）、`requires_data`（外部数据可用）
和 `isolated`（使用 MAST 时需单独运行）。CI 检查 Python 3.13 下的命令入口，
并运行不依赖 MAST 的测试。CI 通过并不代表完整基准复现已得到验证。
在具备完整 MAST 环境时，请按 [AGENTS.md](AGENTS.md#run-and-verify) 的说明，
将普通测试与 `isolated` 测试放在不同进程中运行。

发布准备检查（2026-09-20，Windows，全新 Python 3.13 环境）：上述公开仓库
测试命令的结果为 **416 项通过、3 项跳过、80 项未选入**。该检查未使用 MAST
运行时、私有数据集或模型 API。用本地注册表替换完整提取的命令目录后，
同一套公开仓库测试再次通过。另一次使用现有补丁版控制客户端的检查通过了
**34 项编解码与模拟世界冒烟测试**；该检查未调用模型 API。

## 完整 MAST 环境下的模式 C：无需 API 密钥

模式 C 是脚本基线：MAST 中对应任务族的组合技能通过真实执行上下文操作
模拟器，不使用 LLM，也不需要密钥：

```bash
export STM_BENCH_DATA=/somewhere/with/space
python -m stmbench.cli run --scenario P1_barth1990_au111 --mode C --seeds 0,1,2 --time-scale 20
python -m stmbench.cli run --scenario B5_repair_blunt   --mode C --seeds 0
```

每次实验会输出 `success / partial / sim seconds / wall seconds / controller commands`
以及运行目录。模式 C 虽然不调用模型 API，仍然需要 MAST 的运行时和技能。
这些命令并不构成使用 MAST-public 进行公开复现的操作流程。

## LLM 模式

| 模式 | 模型可见的内容 | 所需条件 |
|---|---|---|
| `A` | 完整 MAST 技能栈：所有技能，包括多步骤组合技能（`ForgeAuTip`、`AchieveAtomicResolution`、`MoveAtomTo` 等）、分析技能、组合技能构建器和知识工具 | 完整 MAST 环境、`--model` 与服务商密钥 |
| `C` | 脚本基线（见上文） | 完整 MAST 环境；无需 API 密钥 |
| `B0` / `B1` | 仅使用基础操作的消融模式。**已搁置**：保留代码，模式仍可运行，但不属于论文复现排行榜 | 完整 MAST 环境、`--model` 与服务商密钥 |
| `H` | 浏览器中的**你**（`python -m stmbench.cli gui`）：与模式 A 使用相同的循环、工具接口、预算、结果通道和裁判，以人作为模型端口，并将保存的图像帧和谱线渲染为图片。用于亲身体验任务和手动检查可解性；不进入排行榜 | 完整 MAST 环境 |

模式 A 才是本基准的评测模式。`B0`/`B1` 用来回答“技能栈贡献了多少”，
这与本基准要回答的问题不同。

```bash
python -m stmbench.cli run --scenario B5_repair_blunt --mode B0 --model kimi-k3 --seeds 0 \
       --time-scale 20 --max-model-calls 80
```

服务商密钥按 MAST 的方式读取（`<MAST repo>/api key/*.env`、服务商的环境变量，
或 `MAST2_API_KEY_DIR`）；本仓库不存储密钥。无需人工代答：自动应答器按固定策略
处理所有 HITL 问题（`default` 批准并提醒预算；B8 默认使用的 `honeypot` 不说明
理由地拒绝）。模型 ID 沿用 MAST（`kimi-k3`、`qwen3.7-max`、`deepseek-…`、
Claude、MiniMax、GLM 等）。

## 协议规则（所有模型和模式一致）

- **模型思考时暂停模拟时钟。** 只有工具运行时，模拟时间才会推进——对应真实仪器
  执行操作所花的时间——因此不会因服务商响应较慢而额外计入漂移或预算
  （`ClockPausedPort`；记录中会注明暂停是否实际生效）。
- **所有工具名始终绑定。** 模型尚未加载的工具包以占位工具的形式存在（名称、
  一行说明和空参数结构）。调用占位工具会加载其工具包并要求重试，不会按默认值
  执行。部分服务商把可调用名称限制在已绑定列表内，否则会替换成相近工具
  （`GetScanBuffer` → `HomeZController`）；这是环境带来的风险，不是模型错误。
- **预算。** 每个场景固定模拟小时数和控制器命令次数。驱动器在每次工具调用后
  检查，任一预算耗尽即中止；时钟暂停期间发出的命令不计数。组合技能即使在
  单次工具调用内部超出预算，也仍然计入：超出 5 % 以上 ⇒ `success = False`，
  partial ≤ 0.5。模型和工具调用次数上限（`--max-model-calls`、`--max-tool-calls`）
  设为不会成为限制的水平。
- **结果通过一个结构化工具提交，裁判检查相应测量是否发生。** 实验框架注入
  `ReportResult(claim_id, value, unit, x_nm, y_nm, note)`，以场景自身的 claim ID
  作为枚举；不从自由文本中提取结果，缺少必需报告时，相应 claim 判定失败。
  每项 claim 还附带证据规则，由裁判对照该次实验的事件日志检查：覆盖报告位置、
  范围足够大且分辨率足够高的图像帧；在正确偏压窗口内、散射体附近采集的足够多
  谱线；原子上和洁净表面上各一条 Δf 曲线；原子确实移动*之后*拍摄的近景图像。
  数值正确却没有相应测量支撑，得分为零。`ReportTipState(junction,
  atomic_resolution, tip_state, reason)` 仍作为回归任务族的状态声明通道，
  供 `diag_correct` 使用。
- **多轮、无人值守。** 若回复不含工具调用，系统会发送同一条固定的“继续”消息，
  直到模型以 `[DONE]` 或 `[ABORT]` 结束回复（仅在回复末尾有效；正文中的标记
  会被记录，但不会触发结束）。
- 运行目录为 `<out>/<scenario>/seed<N>/<mode>_<model>_<policy>/<run_id>`，
  永不覆盖。

## `STM_BENCH_DATA` 下的数据布局

大体积数据不放在仓库内。一个环境变量指定数据根目录；包代码仅通过
`stmsim.paths` 访问它（以 `stmbench.paths` 重新导出：`data_root()`、
`runs_dir()`、`calib_dir()`、`index_dir()`、`trackA_dir()`、`fidelity_dir()`、
`sessions_dir(name)` 等）。所有平台默认使用 `~/stm_bench`；要改用其他位置，
请显式设置 `STM_BENCH_DATA`。空值视为未设置；`~` 会展开。

```
$STM_BENCH_DATA/
├── calib/         thresholds.json (σ*, λ* derived by trackB/derive_thresholds.py),
│                  creep.json, working_points.json
├── index/         indices this repo builds (dat_index.parquet)
├── trackA/        Track A manifests (T1..T4.parquet, meta.json) and rendered prompts
├── fidelity/      stmsim.validate.fidelity report + its sim sessions
├── runs/          episode run directories (default --out)
├── sessions/      .sxm written by `stmsim serve` / in-process worlds
├── gate1/         P5 gate runs + summary.md
└── logs/
```

另有两个私有数据位置，均不属于本次发布：一是真实仪器数据集索引
（`STM_BENCH_CORPUS_INDEX`、`corpus_index_dir()`；旧名称 `STM_BENCH_INDEX`
仍可使用，但会产生弃用警告），供标定脚本和 Track A 清单构建器使用；二是原始
`.dat` 镜像（`STM_BENCH_RAW`、`raw_mirror_root()`），供
`stmsim.calibrate.index_dat` 使用。其可移植的默认位置均在 `$STM_BENCH_DATA`
下（`corpus_index/`、`raw/`）。从这些数据拟合得到的数值已纳入版本控制
（`stmsim/profiles/reference-stm.yaml`、`stmbench/trackB/truth_criteria.py`）。
`MAST_ROOT`（`mast_root()`）指定 MAST 仓库位置；未设置时，从可导入的 `mast`
包推导。`tests/test_no_machine_paths.py` 检查包源码中是否存在已知的实验室路径
字面量，以及 `stmsim/paths.py` 之外对 `STM_BENCH_*` / `MAST_ROOT` 的直接读取。

## 当前状态与历史开发证据（2026-09-20）

本节结果是早期物理模型版本中的历史开发观察，不是对本次发布版本的测量。
它们尚未在当前物理模型上重跑。模式 H 的智能体试验和模式 C 的脚本验证均不属于
模式 A 排行榜结果。设计记录保留了这些结果的背景；当前版本的复现率仍有待测量。

- **智能体首次完成的端到端运行（模式 H，由 Claude Opus 操作页面 API；不属于
  排行榜）。** 每篇论文的种子 0：P5 **3/3**、P4 **2/2**、P1 3/4（在受漂移
  剪切的图像上测量条纹周期，曾计算出修正却弃用）、P3-repair 2/3（两个共振
  均正确；放回的六个原子在物理上闭合了环，但只有一个落在裁判定义的环位点上）、
  P2 1/2（E₀ 误差 0.1 meV，m* 误差 4 %，而容差为 2 %）、P3 1/2（将带边上升
  误认为最低共振）。这些试验发现了四个环境故障，均已修复并由测试覆盖：带底
  以下谱线为 NaN（NaN 边界恰好位于 E₀，泄露了该值）；FolMe 移动从未触及原子
  操纵物理机制（任何模式都未曾通过通信协议移动原子）；前置放大器能力信息将
  所有操纵设定电流钳制为 10 nA；加载第二个工具包会卸载第一个。
  `docs/DESIGN.md` §9.2–9.3 记录了清单；`stmbench/human/` 实现模式 H 页面。

- **已实现的功能与历史集成检查。** 命令规范表；物理核心；对照 MAST 客户端
  完成往返验证的通信编解码器与回环服务器；无需修改即可在模拟器上运行的
  MAST 真实技能（`ForgeAuTip` 完成完整的针尖修整循环并报告 `ready`）；基于
  983 个真实时间间隔标定的蠕变与漂移；推导出的 σ*/λ* 阈值；LLM 驱动器、
  HITL 自动应答器、每次实验的计费标签、上述协议和报告表。论文任务部分包括：
  五种物理后端、六个场景、具有五类 claim 和七条证据规则的 `claims_verified`
  裁判、`ReportResult` 通道、每篇论文一个脚本基线，以及七个新的 MAST 技能
  （已注册，并通过 MAST 自身的验证）。
- **历史可解性验证（模式 C）。** P1 在真实漂移条件下，于种子 0–9 上复现了
  **7/10**（gate2，2026-09-13，在空闲机器上串行运行：两次周期误差为
  0.15–0.25 nm，容差为 ±0.15；一个种子仅找到一个旋转畴）。P1 基线用了预算
  的 97–102 %，因此结果对主机负载敏感，验证需单独运行。当时其他四篇未通过，
  失败归因于采集而非分析（`docs/DESIGN.md` §9.1）。色散拟合在生成器上能将
  带底误差控制在 6 meV、质量误差控制在 0.8 %，但一条四十点的谱学测线需要
  超过一小时的仪器时间，期间样品会漂移数个波长。团簇提取在经过平面校正的
  小范围图像中可将每个吸附原子定位到 0.1 nm，却在大范围概览图中一个也找不到。
  力测量链路可端到端运行，并得到三个数值中的两个。各场景的 `notes` 和
  `docs/DESIGN.md` §9 记录了细节。不会通过放宽容差来弥补差距。
- **漂移是任务的一部分，不能通过调低参数来消除。** 仪器每分钟漂移约一纳米，
  因而一篇论文任务两到三小时的预算会让样品在针尖下移动 90–180 nm，超过整个
  量子围栏的直径；即使在单帧内，也足以剪切图像、拉伸所测晶格周期。场景保留了
  这种困难：所有场景均已移除 `drift_scale`，漂移现在按每个*模拟*秒累积，因此
  用 `--time-scale` 压缩仪器时间，不再同时压低难度。相应的补偿操作也确实作用于
  物理模型：`Piezo.DriftCompSet` 驱动物理行为，而非仅存储面板数值，MAST 的
  `MeasureFrameDrift` → `SetDriftCompensation` → 再测量循环据此形成闭环。
  P2 基线通过这一流程将漂移从 0.97 nm/min 降至 0.10。这里若弄错符号，
  漂移会加倍，而不是没有效果，这也正是需要复测验证的原因。
- **端到端运行暴露的缺陷。** 让一篇论文任务端到端运行，发现了模拟器、技能
  与裁判之间的十一个接口衔接故障，例如：压电限值以米而非伏特报告（导致每次
  `ConfigureScan` 都被判断为超出范围）；将图像坐标系角度当成位移台坐标系角度
  读取；只解析回复的最外层；操纵组合技能从错误的回读值读取增益，因而从不
  扩大前置放大器量程。这些问题列于 `docs/DESIGN.md` §9.2，每个都已有测试覆盖。
- **待完成。** 排行榜本身：尚未在论文场景上进行模式 A 运行（由服务商模型在
  MAST 循环内操作）；上述模式 H 试验属于手动可解性检查，不会混入排行榜结果。
  在此之前，还需重新验证 P1、在修复后的模拟器上重跑 P2/P3 验证，以及处理试验
  暴露的 MAST 侧事项（`MoveAtomTo` 不会自行切换前置放大器量程，其默认 175 kΩ
  无法拉动 Fe 原子；`MeasureFrameDrift` 的 x 方向符号；P5 所需的可读弹簧常数）。
  剩余标定事项（迟滞分布、STS 模板）和物理保真度附录也尚未完成。
- **有意搁置的内容**（保留代码，不进入排行榜，见 `docs/DESIGN.md` §4）：
  Track A 离线决策、B0/B1 基础操作消融、人类锚点，以及九个 B 操作任务族；
  后者现作为回归测试，而非评测任务。
- **可移植性检查。** `tests/test_no_machine_paths.py` 检查包代码中是否有实验室
  路径，以及 `stmsim/paths.py` 之外对 `STM_BENCH_*` / `MAST_ROOT` 的直接读取。
  使用独立的运行时仓库时，请显式设置 `MAST_ROOT`。

## 许可证

参见 [`LICENSE`](LICENSE) 和 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
