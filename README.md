# MyCChessRL

本仓库汇总「象棋小巫师」**XQWL06** 中的局面与棋规逻辑、**pybind11** 扩展、与 **MyElephant / icyElephant** 对齐的 **PyTorch 两阶段策略网络**、**并行环境 PPO 自对弈**示意实现，以及自 **MyElephant** 移植的 **Flask 网页对弈 UI**（人类 vs 纯网络）。

## 工作区中其它目录

| 目录 | 作用 |
|------|------|
| `象棋小巫师/` | 原始 XQWL06.CPP 源码与文档 |
| `pybind11-master/` | CMake 默认识别的 pybind11 源码路径（与 `MyCChessRL/cpp/CMakeLists.txt` 相对） |
| `MyElephant/` | 特征工程、网页对弈、训练代码来源 |
| `icyElephant/` | 早期 notebook 流程参考（本仓库网络结构以 MyElephant `SuccessorPolicy` 为准） |

## C++ 扩展 `xqwl_core`

- **抽离内容**：自 `象棋小巫师/XQWL06.CPP` 第 64–1069 行（常量、Zobrist、局面结构、`GenerateMoves` / `LegalMove` / `MakeMove` / `RepStatus` / `RepValue` / `IsMate` 等），去掉 Windows / 搜索 / UI；**自然限着**与原版 UI 一致：`nMoveNum > 100` 判和。
- **重复与「长打」**：与原版相同的 `RepStatus` / `RepValue` 与 `BAN_VALUE` 刻度（界面文案称「长打」；实现上基于无吃子回退链上的 **反复将军** 标志位，与商业棋规中的「长捉」细则并不完全等价）。
- **生成文件**：`cpp/xqwl_extract.inc` 可由同路径下源文件重新截取（见下节）。

### 编译（需本机已安装 CMake、C++17 编译器、Python 开发头文件）

```powershell
cd MyCChessRL\cpp\build
cmake .. -DPython_EXECUTABLE=(Get-Command python).Source
cmake --build . --config Release
```

将生成的 `xqwl_core.cp310-win_amd64.pyd`（名称随 Python 版本变化）复制到 `mycchess_rl` 包同级的 `site-packages` 或把 `build/Release` 加入 `PYTHONPATH`。未编译时 Python 端自动回退到 `cchess` 的 `GamePlay` 规则（重复/限着较弱）。

### 重新生成 `xqwl_extract.inc`

若升级了 `XQWL06.CPP`，可在 PowerShell 中执行：

```powershell
$lines = Get-Content "..\象棋小巫师\XQWL06.CPP" -Encoding UTF8
$lines[63..1068] | Set-Content "MyCChessRL\cpp\xqwl_extract.inc" -Encoding UTF8
```

## Python 包安装

```powershell
cd MyCChessRL
pip install -e .
pip install -e ".[play]"   # 网页对弈需要 Flask
```

顶层 **`cchess`** 与 **`mycchess_rl`** 并列安装，以保持与原版相同的 `import cchess`。

## 模型与特征

- **网络**：`mycchess_rl.model.SuccessorPolicy` — 与 MyElephant `policy_torch` 相同：`stem` + `num_res_layers` 个 `ResBlock` + 全局池化；**`head_src` / `head_dst` / `value_head(3 类)`**。
- **输入平面**：自 MyElephant 拷贝的 `mycchess_rl.chess`（`encode_model_planes`：7 路有符号子力 + 11 路理据 + `plane_extras` 等），通道数 **`POLICY_SELECT_IN_CHANNELS`**。
- **推理辅助**：`mycchess_rl.policy_inference`（贪心两阶段、批采样、价值期望、旧策略 `log π`）。

## PPO 与并行环境

- **环境**：`mycchess_rl.vec_env.ParallelXiangqiVecEnv` — 每步对所有槽位并行走一步；终局槽位在 `reset_finished` 中重置。
- **训练入口**：`python -m mycchess_rl.train_ppo`（或 `mycchess-train-ppo`）。当前实现为**示意**（短 horizon、简化回报与数值稳定裁剪），便于在此基础上加长 rollout、加熵/学习率调度、分离 critic 等。

主要参数：`--n-env`、`--steps`、`--updates`、`--checkpoint`、`--no-cpp`。

## 网页对弈

```powershell
mycchess-play-web --checkpoint path\to\best.pt --host 0.0.0.0 --port 8765
```

红/黑可选「人类」「纯网络」（贪心两阶段）。规则默认走 **XQWL C++**（若扩展已安装）。

## 许可证

原 XQWL、cchess、MyElephant 各自版权与许可证请以原项目为准；本仓库新增代码见 `LICENSE`。
