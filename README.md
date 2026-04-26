# MyCChessRL

中国象棋强化学习实践：**规则与合法着法仅通过本地编译的 `xqwl_core`（象棋小巫师 XQWL06 核心）提供**，不依赖 `cchess` 或其它 Python 棋规库。神经网络为与 MyElephant 一致的 **两阶段 ICCS 策略 + 行棋方三分类价值头**；特征平面在纯 NumPy 路径下由当前 FEN 与 **当前方合法 ICCS 列表** 计算。

## 依赖

- **Python**：≥ 3.10  
- **NumPy、PyTorch**：见 `requirements.txt` 或与 `pyproject.toml` 同步。  
- **`xqwl_core`**：C++17 扩展（pybind11），**必须**自行编译并加入 `PYTHONPATH` 或安装到当前环境。训练、并行环境与对弈等入口在导入 `mycchess_rl.xqwl_state` 时需要该模块；仅使用 `mycchess_rl.chess` 等纯 NumPy 子模块可不装。  
- **网页对弈**：`flask`（`pip install -r requirements.txt` 或 `pip install -e ".[play]"`）。

```bash
pip install -r requirements.txt
pip install -e .
```

## 编译 `xqwl_core`

源码抽离自 `象棋小巫师/XQWL06.CPP`（`cpp/xqwl_extract.inc`），CMake 默认使用工作区内的 `pybind11-master`。

```text
cd MyCChessRL/cpp/build
cmake .. -DPython_EXECUTABLE=python
cmake --build . --config Release
```

将生成的 `xqwl_core*.pyd` / `xqwl_core*.so` 放到可被 Python 导入的路径。

`Position` 提供 `reset`、`legal_moves_iccs`、`make_move_iccs`、`fen`、`in_check`、`terminal_kind`、`copy`（局面快照）等接口。

## 项目结构（概要）

| 路径 | 说明 |
|------|------|
| `cpp/` | `xqwl_core` CMake 与绑定 |
| `mycchess_rl/xqwl_state.py` | `XqwlGameState`：唯一规则入口 |
| `mycchess_rl/chess/` | 平面编码（无 cchess） |
| `mycchess_rl/model.py` | `SuccessorPolicy` 与 checkpoint 加载 |
| `mycchess_rl/policy_inference.py` | 贪心 / 批采样 / 价值 / PPO 用 `log π` |
| `mycchess_rl/vec_env.py` | 并行环境步进 |
| `mycchess_rl/train_ppo.py` | PPO 示意训练 |
| `mycchess_rl/play_web.py` | Flask 网页对弈 |

## 训练与对弈

```bash
python -m mycchess_rl.train_ppo --n-env 32 --steps 64 --updates 100
mycchess-play-web --checkpoint path/to.pt --host 0.0.0.0 --port 8765
```

## 特征说明

部分「对方着法并集 / 对方吃子目标」等平面在无对方独立引擎枚举时填 **零**，通道维数仍为 **7 + 11 + 47**，与旧 checkpoint 形状兼容；若需完全复刻 MyElephant 数值，可后续在 `xqwl_core` 中增加「指定行棋方生成合法着」接口再接通。

## 许可证

原 XQWL、MyElephant 等各自版权与许可证以原项目为准；本仓库见 `LICENSE`。
