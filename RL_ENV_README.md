# 赛车强化学习环境使用说明

## 概述

这是一个符合 OpenAI Gym/Gymnasium 接口的赛车强化学习环境，可以用于训练强化学习模型。

## 安装依赖

```bash
# 基础依赖（如果还没有安装）
pip install numpy pygame PyOpenGL

# 强化学习库（推荐使用 stable-baselines3）
pip install gymnasium stable-baselines3

# 或者使用旧版 gym
pip install gym stable-baselines3
```

## 项目结构

```
RacingCar/
├─ racingcar/                 # 核心包
│  ├─ __init__.py
│  ├─ config.py
│  ├─ utils.py
│  ├─ game.py
│  ├─ racing_env.py
│  └─ entities/               # 实体定义
│     ├─ car.py
│     ├─ track.py
│     ├─ obstacle.py
│     └─ environment.py
├─ train_rl.py                # 训练脚本
├─ visualize_model.py         # 模型可视化
├─ main.py                    # 游戏入口
└─ RL_ENV_README.md           # 文档
```

## 环境特性

### 状态空间（观察）

- 车辆位置 (x, z) - 归一化
- 车辆角度 - 归一化
- 车辆速度 - 归一化
- 到赛道中心线的距离 - 归一化
- 车辆到赛道中心线的方向 (x, z) - 归一化
- 前方障碍物距离（5个方向的射线检测） - 归一化

总共 12 个特征

### 动作空间

离散动作空间，7 个动作：

- 0: 无动作
- 1: 加速
- 2: 减速
- 3: 左转
- 4: 右转
- 5: 加速+左转
- 6: 加速+右转

### 奖励函数

- 每秒存活: -0.3
- 碰撞惩罚: -100
- 出界惩罚: -50
- 速度奖励: 速度 * 0.1
- 保持在赛道上奖励: (1 - 距离/ROAD_WIDTH) * 0.5

## 使用方法

### 1. 基本使用

```python
from racingcar.racing_env import RacingEnv

# 创建环境（不渲染，用于训练）
env = RacingEnv(render_mode=None)

# 或者创建带渲染的环境（用于测试）
env = RacingEnv(render_mode='human')

# 重置环境
obs, info = env.reset()

# 执行动作
action = env.action_space.sample()  # 随机动作
obs, reward, done, truncated, info = env.step(action)

# 渲染（如果 render_mode='human'）
env.render()

# 关闭环境
env.close()
```

### 2. 使用 Stable-Baselines3 训练

#### 使用 PPO 算法

```python
from stable_baselines3 import PPO
from racingcar.racing_env import RacingEnv

env = RacingEnv(render_mode=None)

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)
model.save("ppo_racing_car")

# 测试模型
env = RacingEnv(render_mode='human')
obs, info = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done or truncated:
        obs, info = env.reset()
```

#### 使用 DQN 算法

```python
from stable_baselines3 import DQN
from racingcar.racing_env import RacingEnv

env = RacingEnv(render_mode=None)

model = DQN('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)
model.save("dqn_racing_car")
```

### 3. 使用训练脚本

```bash
# 使用 PPO 训练
python train_rl.py ppo

# 使用 DQN 训练
python train_rl.py dqn

# 运行随机策略演示
python train_rl.py random
```

### 4. 可视化已训练的模型

#### 方法 1: 使用 train_rl.py

```bash
# 可视化 PPO 模型（默认 5 个回合）
python train_rl.py visualize ppo_racing_car ppo 5

# 可视化 DQN 模型（3 个回合）
python train_rl.py visualize dqn_racing_car dqn 3

# 自动检测算法类型
python train_rl.py visualize ppo_racing_car
```

#### 方法 2: 使用独立的可视化脚本

```bash
# 使用独立的可视化脚本（推荐）
python visualize_model.py ppo_racing_car

# 指定算法和回合数
python visualize_model.py ppo_racing_car ppo 10

# 自动检测算法
python visualize_model.py dqn_racing_car auto 5
```

#### 可视化功能特性

- **实时渲染**: 3D 可视化环境，第三人称视角
- **统计信息**: 显示每个回合的奖励、步数、结束原因
- **动作统计**: 显示每个动作的使用频率
- **总体统计**: 所有回合的平均值、标准差、最大值、最小值
- **交互控制**: 按 ESC 键可以提前退出
- **多回合运行**: 可以连续运行多个回合进行性能评估

## 自定义环境

你可以通过修改 `config.py` 来调整环境参数：

- `CAR_BASE_SPEED`: 车辆基础速度
- `TURN_SPEED`: 转向速度
- `ROAD_WIDTH`: 道路宽度
- `OBSTACLE_COUNT`: 障碍物数量
- `TRACK_POINTS`: 赛道控制点数量

## 注意事项

1. **渲染模式**: 训练时建议使用 `render_mode=None` 以提高训练速度
2. **观察归一化**: 所有观察值都已归一化，但范围可能超出 [-1, 1]
3. **奖励调优**: 可以根据训练效果调整 `racing_env.py` 中的奖励函数
4. **性能优化**: 如果训练速度慢，可以减少 `OBSTACLE_COUNT` 或简化射线检测

## 示例输出

训练过程中，你会看到类似以下的输出：

```
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
```

## 故障排除

1. **ImportError**: 确保安装了所有依赖
2. **OpenGL 错误**: 确保系统支持 OpenGL（主要用于渲染）
3. **训练不收敛**: 尝试调整学习率、奖励函数或网络结构
