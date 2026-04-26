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
mycchess-play-web --checkpoint path/to.pt --host 0.0.0.0 --port 8080
```

训练脚本会按轮打印 **rollout / 优化耗时、样本数、GAE 统计、分项损失、熵、importance ratio、clip 比例、近似 KL、梯度范数、CUDA 显存** 等；`--log-every N` 为每 N 轮打一次，`--log-file` 同步写入文件。`--rollout-log-every`（默认 32）在单轮 rollout 内输出进度，避免首轮长时间无输出。

**数据准备并行**：`mycchess_rl/encode_parallel.py` 用 **常驻进程池**（`spawn` 上下文，与主进程 CUDA 共存更安全）对局面 **14 平面编码**（默认 `min(8, CPU核数)`，可用 `--encode-workers` 覆盖；`1` 为当前进程内顺序编码）。采样阶段对 **整批** 计算 `head_dst`，不再对每个环境单独做一次小矩阵乘。

默认训练超参面向 **约 24 核 CPU、64GB 内存、单卡 A100 40GB**（例如 `--n-env 384 --steps 192 --updates 800`）；价值估计与 rollout 旧对数概率均尽量 **批前向** 以提高 GPU 利用率。优化阶段若整批前向（样本数 ≈ `n_env × steps`）会占满激活显存，可用 **`--ppo-mini-batch`**（默认 4096）按小批 **梯度累积** 做 PPO 反向，峰值显存随该值近似线性变化；仍 OOM 时再调小 `--ppo-mini-batch`、`--n-env` 或 `--steps`。

## 特征说明

ResNet 茎输入为 **14×10×9**（与 icyElephant 数据管线一致）。icy 原版第二阶段曾用 **15 路**（14 棋子 + 1 路起点掩码）喂第二个卷积塔；本仓库仍用 **单塔 trunk**，仅在全连接 `head_dst` 处拼接起点 one-hot，与 icy 的卷积分塔不完全相同，但**棋盘侧张量**已与 icy 对齐。曾用 **65 通道** 旧权重与当前 `stem_conv` **不兼容**，需重新训练。

`mycchess-play-web` 支持 `--workers`（默认 1）；多 worker 时每个进程独立内存，**不宜**与单进程共享会话的用法混用。

## 许可证

原 XQWL、MyElephant 等各自版权与许可证以原项目为准；本仓库见 `LICENSE`。`cpp/pybind11-master/` 为 pybind11 项目源码，以该目录内 `LICENSE` 为准。
