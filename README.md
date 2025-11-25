# Racing Car RL Agent (PPO)

这是一个基于强化学习（Reinforcement Learning）的赛车自动驾驶项目。项目使用了 **Stable Baselines 3** 库中的 **PPO (Proximal Policy Optimization)** 算法来训练赛车智能体，环境基于 Pygame 构建。

项目支持多核 CPU 并行训练、模型断点保存、断点续训以及可视化评估。

## 📂 文件结构

  * `main.py`: 游戏手动运行入口（用于人类玩家游玩或测试游戏逻辑）。
  * `train_rl.py`: 强化学习的核心脚本，包含训练、微调、可视化和随机测试功能。
  * `racingcar/`: 游戏环境包（包含 `RacingEnv` 和 `Game` 类）。
  * `checkpoints/`: 训练过程中自动保存的模型检查点。
  * `logs/`: Tensorboard 日志文件。

## 🛠️ 安装依赖

请确保安装了 Python 3.10+，并安装以下核心依赖库：

```bash
pip install numpy torch stable-baselines3 pygame tensorboard
```
或者
```bash
conda create --name RacingCar python=3.10
pip install -r requirments
```
## 🚀 快速开始

本项目主要通过 `train_rl.py` 进行管理，该脚本使用命令行参数（CLI）来控制不同的模式。

### 1\. 从头开始训练 (Train)

使用 `train` 命令启动新的训练会话。默认使用 CPU 并行环境加速采样。

```bash
# 默认训练 1000万步
python train_rl.py train

# 自定义训练步数和保存频率
python train_rl.py train --timesteps 200000000 --save_freq 200000
```

  * **参数说明**:
      * `--timesteps`: 总训练步数 (默认: 10,000,000)
      * `--save_freq`: 保存模型的频率 (默认: 100,000 步保存一次)
  * **输出**: 模型将保存在 `./checkpoints/ppo/` 目录下，最终模型保存为 `ppo_racing_car_final.zip`。

### 2\. 断点续训 (Continue Training)

如果训练中断或想基于已有模型继续微调，使用 `continue` 命令。

```bash
# 加载 checkpoints/ppo/model_500000.zip 并继续训练
python train_rl.py continue ./checkpoints/ppo/ppo_racing_car_500000

# 指定继续训练的步数
python train_rl.py continue ppo_racing_car_final --timesteps 2000000
```

  * **注意**: `model_path` 参数不需要加 `.zip` 后缀。

### 3\. 模型可视化 (Visualize)

查看训练好的模型在环境中的实际表现（渲染模式）。

```bash
# 运行 5 个回合进行观察
python train_rl.py viz ppo_racing_car_final --episodes 5
```

  * **操作**: 在可视化窗口中，按 `ESC` 键或点击关闭按钮可提前退出。

### 4\. 随机策略测试 (Random Demo)

测试环境是否正常工作，或者查看随机动作下的表现。

```bash
python train_rl.py random
```

### 5\. 手动游玩 (Human Play)

如果你想自己操作赛车体验游戏：

```bash
python main.py
```

## ⚙️ 关键配置 (train\_rl.py)

如果需要调整训练超参数，请直接修改 `train_rl.py` 顶部的全局变量或 PPO 初始化参数(建议 n_envs * n_step = 4096)：

  * **并行环境数**:
    ```python
    N_ENVS = 8  # 建议设置为你的 CPU 物理核心数
    ```
  * **PPO 参数**:
    当前配置针对 CPU 训练进行了优化：
      * `device="cpu"`: 强制使用 CPU。
      * `batch_size=256`: 适合 CPU 推理的批次大小。
      * `n_steps=128`: 每个环境采样的步数。

## 📈 监控训练

训练过程中会生成 Tensorboard 日志。在训练运行时，可以通过以下命令查看训练曲线：

```bash
tensorboard --logdir ./logs/
```

然后在浏览器访问 `http://localhost:6006`。