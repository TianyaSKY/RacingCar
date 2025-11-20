# RacingCar - 赛车强化学习环境

一个基于 Pygame 和 OpenGL 的 3D 赛车强化学习环境，符合 OpenAI Gymnasium 接口标准，支持使用 Stable-Baselines3 进行强化学习训练。

## ✨ 特性

- 🎮 **3D 可视化环境**：使用 OpenGL 渲染的 3D 赛车环境，支持实时渲染
- 🏎️ **完整物理模拟**：包含车辆运动、碰撞检测、赛道边界等物理特性
- 🧠 **强化学习支持**：兼容 Gymnasium 接口，支持 PPO、DQN 等主流算法
- 📊 **训练工具**：内置训练脚本，支持模型训练、继续训练、可视化等功能
- 💾 **检查点系统**：自动保存训练检查点，支持断点续训
- 📈 **TensorBoard 集成**：支持训练过程可视化

## 📋 目录

- [安装](#安装)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [环境说明](#环境说明)
- [使用指南](#使用指南)
- [配置说明](#配置说明)
- [故障排除](#故障排除)

## 🚀 安装

### 环境要求

- Python 3.9+
- 支持 OpenGL 的显卡（用于渲染）

### 安装步骤

1. **克隆或下载项目**

```bash
cd RacingCar
```

2. **安装依赖**

```bash
pip install -r requirements.txt
```

## 🎯 快速开始

### 1. 训练模型

使用 PPO 算法训练一个新模型：

```bash
python train_rl.py ppo
```

训练过程会自动：

- 每 10,000 步保存一次检查点到 `./checkpoints/ppo/`
- 记录 TensorBoard 日志到 `./logs/`
- 训练完成后保存最终模型为 `ppo_racing_car.zip`

### 2. 可视化模型

查看训练好的模型表现：

```bash
python train_rl.py visualize ppo_racing_car ppo 5
```

```bash
python train_rl.py visualize .\checkpoints\ppo\ppo_racing_car_100000
```

这将运行 5 个回合，展示模型在环境中的表现。

### 3. 继续训练

基于已有模型继续训练：

```bash
python train_rl.py continue ppo_racing_car ppo 100000 10000
```

参数说明：

- `ppo_racing_car`: 模型路径（不含 .zip 扩展名）
- `ppo`: 算法类型
- `100000`: 继续训练的步数
- `10000`: 检查点保存频率（可选，默认 10000）

### 4. 随机策略演示

测试环境是否正常工作：

```bash
python train_rl.py random
```

## 📁 项目结构

```
RacingCar/
├── racingcar/                 # 核心包
│   ├── __init__.py
│   ├── config.py              # 环境配置参数
│   ├── utils.py               # 工具函数
│   ├── game.py                # 游戏主循环
│   ├── racing_env.py          # Gymnasium 环境实现
│   └── entities/              # 实体定义
│       ├── car.py             # 车辆实体
│       ├── track.py           # 赛道生成
│       ├── obstacle.py        # 障碍物
│       └── environment.py     # 环境渲染
├── train_rl.py                # 训练脚本（主要入口）
├── visualize_model.py         # 模型可视化脚本
├── main.py                    # 游戏入口（手动控制）
├── requirements.txt           # 依赖列表
├── README.md                  # 本文档
└── checkpoints/               # 模型检查点目录（自动创建）
    └── ppo/
        ├── ppo_racing_car_10000.zip
        ├── ppo_racing_car_20000.zip
        └── ...
```

## 🎮 环境说明

### 观察空间（Observation Space）

观察空间为 44 维连续向量，包含：

**基础特征（7 维）：**

- `x, z`: 车辆位置（归一化到 [-1, 1]）
- `angle`: 车辆角度（归一化到 [-1, 1]）
- `speed`: 车辆速度（归一化）
- `track_dist`: 到赛道中心线的距离（归一化）
- `track_dir_x, track_dir_z`: 车辆到赛道中心线的方向向量（归一化）

**射线检测（37 维）：**

- 37 个方向的障碍物距离检测（归一化到 [0, 1]，1 表示无障碍物）

```python
observation_space = Box(low=-inf, high=inf, shape=(44,), dtype=np.float32)
```

### 动作空间（Action Space）

离散动作空间，7 个动作：

| 动作 | 说明        |
| ---- | ----------- |
| 0    | 无动作      |
| 1    | 加速        |
| 2    | 减速        |
| 3    | 左转        |
| 4    | 右转        |
| 5    | 加速 + 左转 |
| 6    | 加速 + 右转 |

```python
action_space = Discrete(7)
```

### 奖励函数

奖励设计鼓励智能体：

- 保持高速行驶
- 保持在赛道上
- 避免碰撞和出界

具体奖励：

- **每步存活惩罚**: -0.5（鼓励快速完成）
- **速度奖励**: `速度 × 0.8`（鼓励高速）
- **赛道保持奖励**: `(1 - 距离/ROAD_WIDTH) × 0.05`（鼓励在赛道中心）
- **碰撞惩罚**: -100
- **出界惩罚**: -50

### 终止条件

回合在以下情况结束：

- 车辆与障碍物碰撞
- 车辆离开赛道超过容忍范围
- 达到最大步数（2000 步）

## 📖 使用指南

### 训练命令

```bash
# 训练 PPO 模型（默认 1,000,000 步）
python train_rl.py ppo

# 运行随机策略演示
python train_rl.py random
```

### 可视化命令

```bash
# 基本用法：可视化 5 个回合
python train_rl.py visualize ppo_racing_car ppo 5

# 指定算法和回合数
python train_rl.py visualize ppo_racing_car ppo 10

# 使用默认参数（自动检测算法，5 个回合）
python train_rl.py visualize ppo_racing_car
```

**可视化功能：**

- 实时 3D 渲染，第三人称视角
- 显示步数、速度、到赛道距离等信息
- 回合结束后显示统计信息
- 按 `ESC` 键可提前退出

### 继续训练命令

```bash
# 基本用法：继续训练 100,000 步
python train_rl.py continue ppo_racing_car ppo 100000

# 指定检查点保存频率
python train_rl.py continue ppo_racing_car ppo 100000 20000

# 使用 resume 命令（与 continue 相同）
python train_rl.py resume ppo_racing_car ppo 50000
```

**继续训练特性：**

- 自动加载已有模型权重
- 保持原有训练配置（学习率、网络结构等）
- 累计训练步数（不重置计数器）
- 自动保存检查点和最终模型

### 查看训练日志

使用 TensorBoard 查看训练过程：

```bash
tensorboard --logdir ./logs/
```

然后在浏览器中打开 `http://localhost:6006`

## ⚙️ 配置说明

可以通过修改 `racingcar/config.py` 来调整环境参数：

### 物理参数

```python
CAR_BASE_SPEED = 0.8      # 车辆基础速度
TURN_SPEED = 2.5          # 转向速度（度/帧）
ACCELERATION = 0.02       # 加速度
FRICTION = 0.96           # 摩擦力系数
```

### 赛道参数

```python
ROAD_WIDTH = 10.0         # 道路宽度
OFF_ROAD_TOLERANCE = 2.0  # 出界容忍距离
TRACK_POINTS = 20         # 赛道控制点数量
TRACK_RADIUS = 150        # 赛道半径
TRACK_VARIANCE = 60       # 赛道扭曲程度
```

### 障碍物参数

```python
OBSTACLE_COUNT = 15       # 障碍物数量
```

### 训练参数

在 `train_rl.py` 中可以调整训练超参数：

```python
model = PPO(
    'MlpPolicy',
    env,
    learning_rate=3e-4,   # 学习率
    n_steps=2048,         # 每次更新的步数
    batch_size=64,        # 批次大小
    n_epochs=10,          # 每次更新的轮数
    gamma=0.99,           # 折扣因子
    gae_lambda=0.95,      # GAE 参数
    clip_range=0.2,       # PPO 裁剪范围
)
```

## 🔧 故障排除

### 常见问题

**1. ImportError: No module named 'stable_baselines3'**

```bash
pip install stable-baselines3
```

**2. OpenGL 相关错误**

确保系统支持 OpenGL。如果不需要渲染，可以在训练时使用 `render_mode=None`。

**3. 观察空间维度不匹配**

如果加载旧模型时出现维度错误，可能是环境配置发生了变化。解决方案：

- 使用与训练时相同的环境配置
- 或者重新训练模型

**4. 训练速度慢**

- 减少 `OBSTACLE_COUNT`（障碍物数量）
- 减少射线检测数量（修改 `ray_angles`）
- 使用 `render_mode=None` 进行训练
- 增加并行环境数量（`n_envs`）

**5. 模型不收敛**

尝试：

- 调整学习率（降低或提高）
- 修改奖励函数权重
- 增加训练步数
- 调整网络结构

### 性能优化建议

1. **训练时关闭渲染**：使用 `render_mode=None`
2. **使用向量化环境**：`make_vec_env` 可以并行运行多个环境
3. **调整检查点频率**：减少保存频率可以提高训练速度
4. **使用 GPU**：Stable-Baselines3 会自动使用 GPU（如果可用）

## 📊 训练输出示例

训练过程中会看到类似输出：

```
开始训练 PPO 模型...
----------------------------------
| rollout/           |          |
|    ep_len_mean     |   250    |
|    ep_rew_mean     |  15.2    |
| time/              |          |
|    fps             |  120     |
|    iterations      |  10      |
|    time_elapsed    |  85      |
|    total_timesteps |  10240   |
----------------------------------
[Checkpoint] 已保存模型至 ./checkpoints/ppo/ppo_racing_car_10000.zip
```

可视化时会显示：

```
=== 回合 1/5 ===
回合 1 结束:
  总奖励: 125.50
  步数: 342
  结束原因: 完成

==================================================
可视化统计:
==================================================
总回合数: 5
平均奖励: 118.30 ± 15.20
最高奖励: 145.60
最低奖励: 95.40
平均步数: 325.40 ± 28.50
最长步数: 380
最短步数: 280
==================================================
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目采用 MIT 许可证。

## 🙏 致谢

- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - 强化学习算法库
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) - 强化学习环境标准
- [Pygame](https://www.pygame.org/) - 游戏开发库
- [PyOpenGL](http://pyopengl.sourceforge.net/) - OpenGL Python 绑定

---

**Happy Training! 🏎️💨**
