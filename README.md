# MyCChessRL

中国象棋强化学习实践：**规则与合法着法仅通过本地编译的 `xqwl_core`（象棋小巫师 XQWL06 核心）提供**，不依赖 `cchess` 或其它 Python 棋规库。神经网络为 **`JointPolicyValueNet`：对有序合法着法列表的联合 softmax（槽位宽度见 `POLICY_MAX_LEGAL_MOVES`）+ `tanh` 缩放的标量价值**；**根输入 14 路棋子平面**与 [icyElephant](https://github.com/bupticybee/icyElephant) 的 `game_convert.py` / `gameplay.py` 一致（行棋方子类在前，黑方时垂直翻转），而非旧版 MyElephant 式 7+11+47 融合平面。**与旧「两阶段 src/dst + 三分类价值」权重不兼容，需重新训练。**

## 依赖

- **Python**：≥ 3.10  
- **NumPy、PyTorch、xmltodict**：见 `requirements.txt` 或与 `pyproject.toml` 同步（监督学习与 icyElephant/MyElephant 棋谱解析一致）。  
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
| `mycchess_rl/model.py` | `JointPolicyValueNet` 与 `load_policy_value_for_play` |
| `mycchess_rl/policy_inference.py` | 贪心 / 批采样 / 价值 / PPO 用 `log π`（合法着法联合分布） |
| `mycchess_rl/vec_env.py` | 并行环境步进 |
| `mycchess_rl/train_ppo.py` | PPO 示意训练 |
| `mycchess_rl/train_sl.py` | icyElephant 风格 XML ``.cbf`` 监督学习（联合策略头 + 价值 MSE） |
| `mycchess_rl/sl_data.py` | 棋谱 IterableDataset / DataLoader |
| `mycchess_rl/play_web.py` | Sanic 网页对弈（纯网络 / MCTS） |
| `mycchess_rl/mcts_joint.py` | AlphaZero 式 PUCT + 联合策略先验与 NN 价值 |

## 训练与对弈

```bash
python -m mycchess_rl.train_ppo
python -m mycchess_rl.train_ppo --log-file runs/ppo.log --log-every 5
python -m mycchess_rl.train_ppo --save-dir runs --save-every 100
python -m mycchess_rl.train_ppo --resume --save-dir runs
python -m mycchess_rl.train_ppo --resume --checkpoint runs/weights/upd_000499.pt
mycchess-play-web --checkpoint path/to.pt --host 0.0.0.0 --port 8080
# 网页侧下拉选「MCTS」；PUCT 次数与 c_puct 可调：
mycchess-play-web --checkpoint path/to.pt --mcts-simulations 800 --mcts-c-puct 1.5
```

**监督学习（cbf）**：与 [icyElephant](https://github.com/bupticybee/icyElephant) / MyElephant 相同 **XML ``ChineseChessRecord``** 棋谱；在 ``xqwl_core`` 上回放，对 **排序后的合法 ICCS 槽位** 做交叉熵，对 **行棋方终局**（``RecordResult`` 与 icy 一致：1 红胜 / 2 黑胜 / 3–4 和）做 **MSE 到 ``±value_scale`` / 0**。**当前引擎无 ``set_fen``**，仅加载 **与标准起始局面一致** 的 ``Head/FEN`` 的棋谱；中局起点或规则与 xqwl 不一致的着法会 **静默跳过** 该文件。

**用法**：必须提供 **`--cbf-root`**（或 **`--cbf-manifest`**）。**`--checkpoint`** 可选：不写则脚本在 **`--save-dir`** 下 **自动生成** ``bootstrap.pt``（随机权重 + AdamW 初态，``epoch=-1``）并立刻开始训练；写了则加载该 ``.pt``。若文件是 **本脚本保存的 SL**（含整数 **`epoch`** 字段），会 **同时恢复优化器与 epoch/step**；若是 **PPO 等**（无 ``epoch``），则 **只加载权重**，优化器重新累积。

```bash
python -m mycchess_rl.train_sl --cbf-root /path/to/cbf --save-dir runs
python -m mycchess_rl.train_sl --cbf-root /path/to/cbf --checkpoint runs/best.pt --save-dir runs
python -m mycchess_rl.train_sl --cbf-root /path/to/cbf --checkpoint runs/last.pt --epochs 5 --save-dir runs
python -m mycchess_rl.train_sl --cbf-manifest my_cbfs.txt --epochs 2 --recount-samples
```

**`--epochs`**：本轮再跑多少个 epoch（续 SL 时在已完成的 epoch 之后追加）。

监督学习在 **主进程** 内用 **``xmltodict``** 读棋谱、**不使用 DataLoader**；每 epoch **先扫完整个训练集**（打乱文件顺序、每文件一轮）→ **再扫完整个验证集** → 验证 loss 更优则写 ``best.pt``，且每轮都写 ``last.pt``（YOLO 习惯）。进度用 ``tqdm``；首次会统计 train/val 样本条数（可缓存在 ``save-dir/sl_sample_counts.json``，``--recount-samples`` 强制重算）。显存允许时可增大 ``--batch-size``。字段与 ``play_web`` / PPO 的 ``model`` 块兼容。

``train_ppo`` 会按轮打印 **rollout / 优化耗时、样本数、GAE 统计、分项损失、熵、importance ratio、clip 比例、近似 KL、梯度范数、CUDA 显存** 等；`--log-every N` 为每 N 轮打一次，`--log-file` 同步写入文件。`--rollout-log-every`（默认 32）在单轮 rollout 内输出进度，避免首轮长时间无输出。**`--random-action-prob`**（默认 **0.1**）：每步以该概率用均匀随机合法着替代策略样本，利于非常规局面；设为 **0** 关闭。

**中途存盘（YOLO 风格）**：在 **`--save-dir`** 下每完成一轮 PPO 更新即覆盖 **`last.pt`**；当本轮 **`loss_total`** 低于历史最佳时额外写入 **`best.pt`**。可选 **`--save-every N`**（默认 50）：非 0 时另在 **`save-dir/weights/upd_000049.pt`** 按全局 ``upd`` 保留快照。训练用 checkpoint 含 **`optimizer`**（Adam）与 **`update`**，**`--resume`** 且未指定 **`--checkpoint`** 时默认读 **`save-dir/last.pt`**；也可 **`--checkpoint save-dir/best.pt`** 续训。仅推理可不设 ``--resume``，只读 ``model`` 等字段。

**奖励塑形**：仅两项（**0=关闭该项**）。`--reward-shaping-king`：非终局鼓励落点靠近对方将/帅（接近度 ``prox∈[0,1]``）；若走后对方被应将，再叠加 ``(0.4+0.6·prox)`` 乘同一系数。`--reward-shaping-capture`：吃子时基量 × 子种相对权重（兵卒=1，象士、马、炮、车、将递增，见 ``vec_env`` 内 ``_CAPTURE_MULT``）。将死仍 **+1**；和棋等终局非将死为 **0**。两项全 **0** 时只有稀疏终局奖。

**数据准备 / GPU 占用**：14 路根平面编码计算量很小，**默认 `--encode-backend inline`**（主进程批量 numpy + **一次** H2D），避免 rollout 每步两次大批编码时 **进程池 pickle/IPC** 反压 GPU（表现为 `nvidia-smi` 利用率低、编码 worker 进程 CPU 也低）。可选 `--encode-backend thread` 或 `process`，并配合 `--encode-workers`。**`--rollout-pipeline-groups`**（默认 2）：采样与价值前向在编码阶段把局面拆成两半，主线程与守护线程各编一半再拼批、一次 `trunk`（`eval()` 下与整批一致），叠合 CPU 准备空档；设为 **1** 关闭。rollout 每步仍有 **策略 + 价值** 两次 trunk（状态不同）。若 GPU 仍低，可增大 `--n-env` 或尝试 `torch.compile` 等。

默认训练超参面向 **约 24 核 CPU、64GB 内存、单卡 A100 40GB**（例如 `--n-env 384 --steps 192`；`--updates` 默认 **100000**，可手动 Ctrl+C 早停）；价值估计与 rollout 旧对数概率均尽量 **批前向** 以提高 GPU 利用率。优化阶段若整批前向（样本数 ≈ `n_env × steps`）会占满激活显存，可用 **`--ppo-mini-batch`**（默认 4096）按小批 **梯度累积** 做 PPO 反向，峰值显存随该值近似线性变化；仍 OOM 时再调小 `--ppo-mini-batch`、`--n-env` 或 `--steps`。

训练日志里的 **torch_reserved** 多为 CUDA 分配器缓存（首轮大包峰值后常明显高于 **torch_alloc**），一般不是显存泄漏；每轮结束后脚本会 `del` 大张量并默认 `torch.cuda.empty_cache()`（可用 `--no-cuda-empty-cache-each-update` 关闭）。碎片严重时可试环境变量 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

## 特征说明

ResNet 茎输入为 **14×10×9**（与 icyElephant 数据管线一致）。策略在 trunk 后接 **`policy_head`（宽度 `policy_max_legal`，与局面「排序后的合法 ICCS 列表」对齐）**；价值为 **`value_fc` → `tanh` × `value_scale`**（默认与训练时 `ret` 裁剪量级一致）。**棋盘侧张量**与 icy 对齐。曾用 **65 通道** 或其它旧 head 的权重与当前结构 **不兼容**，需重新训练。

`mycchess-play-web` 支持 `--workers`（默认 1）；多 worker 时每个进程独立内存，**不宜**与单进程共享会话的用法混用。

## 许可证

原 XQWL、MyElephant 等各自版权与许可证以原项目为准；本仓库见 `LICENSE`。`cpp/pybind11-master/` 为 pybind11 项目源码，以该目录内 `LICENSE` 为准。
