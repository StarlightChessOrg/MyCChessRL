# MyCChessRL

中国象棋强化学习实践：**规则与合法着法仅通过本地编译的 `xqwl_core`（象棋小巫师 XQWL06 核心）提供**，不依赖 `cchess` 或其它 Python 棋规库。神经网络为 **两阶段 ICCS 策略 + 行棋方三分类价值头**；**根输入 14 路棋子平面**与 [icyElephant](https://github.com/bupticybee/icyElephant) 的 `game_convert.py` / `gameplay.py` 一致（行棋方子类在前，黑方时垂直翻转），而非旧版 MyElephant 式 7+11+47 融合平面。

## 依赖

- **Python**：≥ 3.10  
- **NumPy、PyTorch**：见 `requirements.txt` 或与 `pyproject.toml` 同步。  
- **`xqwl_core`**：C++17 扩展（pybind11），**必须**自行编译并加入 `PYTHONPATH` 或安装到当前环境。训练、并行环境与对弈等入口在导入 `mycchess_rl.xqwl_state` 时需要该模块；仅使用 `mycchess_rl.chess` 等纯 NumPy 子模块可不装。  
- **网页对弈**：`sanic`（`pip install -r requirements.txt` 或 `pip install -e ".[play]"`）。

```bash
pip install -r requirements.txt
pip install -e .
```

## 编译 `xqwl_core`

源码抽离自 `象棋小巫师/XQWL06.CPP`（`cpp/xqwl_extract.inc`）。**pybind11** 已随仓库放在 **`cpp/pybind11-master/`**（固定 tag **v2.13.6**），CMake **优先** `add_subdirectory` 使用该目录，**无需 pip、无需访问 GitHub** 即可配置。若你删掉了该目录，可再装 `pip install pybind11`，或加 **`-DMYCCHESSRL_FETCH_PYBIND11=ON`** 尝试在线拉取。

```text
cd MyCChessRL/cpp/build
cmake .. -DPython_EXECUTABLE="$(which python)"
cmake --build . --config Release
```

将生成的 `xqwl_core*.pyd` / `xqwl_core*.so` 放到可被 Python 导入的路径。

`Position` 提供 `reset`、`legal_moves_iccs`、`make_move_iccs`、`fen`、`in_check`、`terminal_kind`、`copy`（局面快照）等接口。

## 项目结构（概要）

| 路径 | 说明 |
|------|------|
| `cpp/` | `xqwl_core` CMake 与绑定 |
| `cpp/pybind11-master/` | 随仓库自带的 [pybind11](https://github.com/pybind/pybind11) v2.13.6 源码（BSD 许可证，见该目录 `LICENSE`） |
| `mycchess_rl/xqwl_state.py` | `XqwlGameState`：唯一规则入口 |
| `mycchess_rl/chess/` | 平面编码（无 cchess） |
| `mycchess_rl/model.py` | `SuccessorPolicy` 与 checkpoint 加载 |
| `mycchess_rl/policy_inference.py` | 贪心 / 批采样 / 价值 / PPO 用 `log π` |
| `mycchess_rl/vec_env.py` | 并行环境步进 |
| `mycchess_rl/train_ppo.py` | PPO 示意训练 |
| `mycchess_rl/play_web.py` | Sanic 网页对弈 |

## 训练与对弈

```bash
python -m mycchess_rl.train_ppo
python -m mycchess_rl.train_ppo --log-file runs/ppo.log --log-every 5
python -m mycchess_rl.train_ppo --save-dir runs --save-every 100
mycchess-play-web --checkpoint path/to.pt --host 0.0.0.0 --port 8080
```

训练脚本会按轮打印 **rollout / 优化耗时、样本数、GAE 统计、分项损失、熵、importance ratio、clip 比例、近似 KL、梯度范数、CUDA 显存** 等；`--log-every N` 为每 N 轮打一次，`--log-file` 同步写入文件。`--rollout-log-every`（默认 32）在单轮 rollout 内输出进度，避免首轮长时间无输出。

**中途存盘**：默认 **`--save-dir runs`**，每 **`--save-every`** 轮（默认 **50**）写入 `runs/ppo_upd_000049.pt` 等（完成第 `upd` 轮后，当 `(upd+1)` 整除 `save_every` 时保存）；训练结束再写 **`runs/mycchess_ppo_last.pt`**。`--save-every 0` 则仅写 `last`。checkpoint 内含 `model`、`in_channels`、可选 `update`，与 `--checkpoint` 微调加载格式一致。

**奖励塑形（初期易瞎逛时）**：`--reward-shaping-step`（默认小负数）时间压力；`--reward-shaping-king-prox` 按落点与对方将/帅**接近度**（曼哈顿，归一化到 [0,1]）给微弱奖；`--reward-shaping-check` 在对手**应将**时给奖，并随接近度在 **0.4~1.0** 倍缩放；`--reward-shaping-capture` 为吃子基量，**兵卒=1×**，象士、马、炮、车、将（若出现）权重递增（见 ``vec_env._CAPTURE_MULT``）。将死仍 **+1**。全关：四个参数均 **0**。

**战术微弱塑形**（默认系数 ``reward_patterns.DEFAULT_TACTIC_SHAPING_COEFF``（当前 **0.004**），与 ``--reward-shaping-king-prox`` 同量级；各 ``--reward-shaping-*`` 传 **0** 可单独关闭该项，实现见 ``mycchess_rl/reward_patterns.py``）：`--reward-shaping-ae-shape`（士在九宫、象在己方半场）；`--reward-shaping-double-cannon`（担子炮）；`--reward-shaping-rook-pair`（双车同横线且间距较大）；`--reward-shaping-cross-pawn`（过河卒）；`--reward-shaping-knight-flex`（刚走动为马时，按棋盘**伪合法**马步中「较好」马步数计分）。另有 `--reward-shaping-three-edge`（三子归边：车马炮在对方半场一侧翼 x≤2 或 x≥6 上≥3 枚）、`--reward-shaping-central-cannon`（中炮：己炮在 x=4 且未过河；对方马在对方九宫心且同列时**加倍**）、`--reward-shaping-open-cannon`（空头炮：己炮与对方将同纵线，其间无子弱奖、恰一枚对方子作炮架时满系数）、`--reward-shaping-rook-pin-cannon`（车牵炮：己车与对方车、炮共线且该直/横线上**仅有**这三枚子，近似「车牵无根车炮线」）。将类：`--reward-shaping-opp-king-gate`（对方将在九宫内正交可进空格≤1，将门被堵）、`--reward-shaping-miss-adv-double-rook`（己方士少于 2 且对方两车则**减去**该系数）、`--reward-shaping-double-adv-king-center`（己方恰双士且将在九宫几何中心）、`--reward-shaping-king-near-start`（己将/帅距开局格曼哈顿越近奖越大，除以 10 归一）。将位与九宫与 ``FULL_INIT_FEN`` 经 ``board_view`` 同款翻转一致；若引擎初始 FEN 不同，可后续改为读局面快照。

**数据准备 / GPU 占用**：14 路根平面编码计算量很小，**默认 `--encode-backend inline`**（主进程批量 numpy + **一次** H2D），避免 rollout 每步两次大批编码时 **进程池 pickle/IPC** 反压 GPU（表现为 `nvidia-smi` 利用率低、编码 worker 进程 CPU 也低）。可选 `--encode-backend thread` 或 `process`，并配合 `--encode-workers`。**`--rollout-pipeline-groups`**（默认 2）：采样与价值前向在编码阶段把局面拆成两半，主线程与守护线程各编一半再拼批、一次 `trunk`（`eval()` 下与整批一致），叠合 CPU 准备空档；设为 **1** 关闭。rollout 每步仍有 **策略 + 价值** 两次 trunk（状态不同）。若 GPU 仍低，可增大 `--n-env` 或尝试 `torch.compile` 等。

默认训练超参面向 **约 24 核 CPU、64GB 内存、单卡 A100 40GB**（例如 `--n-env 384 --steps 192 --updates 800`）；价值估计与 rollout 旧对数概率均尽量 **批前向** 以提高 GPU 利用率。优化阶段若整批前向（样本数 ≈ `n_env × steps`）会占满激活显存，可用 **`--ppo-mini-batch`**（默认 4096）按小批 **梯度累积** 做 PPO 反向，峰值显存随该值近似线性变化；仍 OOM 时再调小 `--ppo-mini-batch`、`--n-env` 或 `--steps`。

训练日志里的 **torch_reserved** 多为 CUDA 分配器缓存（首轮大包峰值后常明显高于 **torch_alloc**），一般不是显存泄漏；每轮结束后脚本会 `del` 大张量并默认 `torch.cuda.empty_cache()`（可用 `--no-cuda-empty-cache-each-update` 关闭）。碎片严重时可试环境变量 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

## 特征说明

ResNet 茎输入为 **14×10×9**（与 icyElephant 数据管线一致）。icy 原版第二阶段曾用 **15 路**（14 棋子 + 1 路起点掩码）喂第二个卷积塔；本仓库仍用 **单塔 trunk**，仅在全连接 `head_dst` 处拼接起点 one-hot，与 icy 的卷积分塔不完全相同，但**棋盘侧张量**已与 icy 对齐。曾用 **65 通道** 旧权重与当前 `stem_conv` **不兼容**，需重新训练。

`mycchess-play-web` 支持 `--workers`（默认 1）；多 worker 时每个进程独立内存，**不宜**与单进程共享会话的用法混用。

## 许可证

原 XQWL、MyElephant 等各自版权与许可证以原项目为准；本仓库见 `LICENSE`。`cpp/pybind11-master/` 为 pybind11 项目源码，以该目录内 `LICENSE` 为准。
