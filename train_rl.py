# train_rl.py
"""
强化学习训练脚本示例
使用 Stable-Baselines3 或类似的 RL 库进行训练
"""
import os
import numpy as np
from racingcar.racing_env import RacingEnv
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
SB3_AVAILABLE = True


class PeriodicCheckpointCallback(BaseCallback):
    """每隔固定步数保存一次模型"""

    def __init__(self, save_freq: int, save_dir: str, name_prefix: str = "model", verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_dir = save_dir
        self.name_prefix = name_prefix
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            save_path = os.path.join(
                self.save_dir, f"{self.name_prefix}_{self.num_timesteps}")
            self.model.save(save_path)
            if self.verbose > 0:
                print(f"[Checkpoint] 已保存模型至 {save_path}.zip")
        return True


def train_with_ppo():
    """使用 PPO 算法训练"""

    env = make_vec_env(
        RacingEnv,
        n_envs=8,
        env_kwargs={'render_mode': None},
        vec_env_cls=SubprocVecEnv,
        seed=42
    )

    # 创建 PPO 模型
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log="./logs/",
    )

    # 训练
    print("开始训练 PPO 模型...")
    checkpoint_callback = PeriodicCheckpointCallback(
        save_freq=10_000,
        save_dir="./checkpoints/ppo",
        name_prefix="ppo_racing_car",
        verbose=1,
    )
    model.learn(total_timesteps=500000, callback=checkpoint_callback)

    # 保存模型
    model.save("ppo_racing_car")
    print("模型已保存为 ppo_racing_car")

    # 测试训练好的模型
    test_model(model)

    env.close()


def continue_training(model_path, algorithm='ppo', total_timesteps=100000, 
                     save_freq=10000, save_dir=None, name_prefix=None):
    """
    基于已有模型继续训练
    
    Args:
        model_path: 模型文件路径（不含扩展名）
        algorithm: 算法类型 ('ppo' 或 'dqn')
        total_timesteps: 继续训练的步数
        save_freq: 保存检查点的频率（步数）
        save_dir: 检查点保存目录
        name_prefix: 检查点文件名前缀
    """
    import os
    
    # 检查模型文件是否存在
    if not os.path.exists(f"{model_path}.zip"):
        print(f"错误: 找不到模型文件 {model_path}.zip")
        print("请确保模型文件存在。")
        return
    
    # 加载模型
    print(f"正在加载模型: {model_path}.zip")
    try:
        if algorithm.lower() == 'ppo':
            model = PPO.load(model_path)
        elif algorithm.lower() == 'dqn':
            model = DQN.load(model_path)
        else:
            print(f"错误: 不支持的算法类型 '{algorithm}'")
            return
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    print(f"模型加载成功！")
    print(f"模型已训练步数: {model.num_timesteps}")
    print(f"将继续训练 {total_timesteps} 步...")
    
    # 设置环境（使用与模型训练时相同的环境配置）
    # 注意：如果环境配置发生变化，可能需要重新创建环境
    env = make_vec_env(
        RacingEnv,
        n_envs=2,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={'render_mode': None},
        seed=42
    )
    
    # 设置模型的环境（如果需要）
    model.set_env(env)
    
    # 设置检查点保存参数
    if save_dir is None:
        save_dir = f"./checkpoints/{algorithm.lower()}"
    if name_prefix is None:
        name_prefix = f"{algorithm.lower()}_racing_car"
    
    # 创建检查点回调
    checkpoint_callback = PeriodicCheckpointCallback(
        save_freq=save_freq,
        save_dir=save_dir,
        name_prefix=name_prefix,
        verbose=1,
    )
    
    # 继续训练
    print("\n开始继续训练...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        reset_num_timesteps=False  # 不重置步数计数器，继续累计
    )
    
    # 保存更新后的模型
    model.save(model_path)
    print(f"\n模型已更新并保存为 {model_path}.zip")
    print(f"总训练步数: {model.num_timesteps}")
    
    env.close()


def test_model(model):
    """测试训练好的模型"""
    # 创建带渲染的环境
    env = RacingEnv(render_mode='human')

    obs, info = env.reset()
    total_reward = 0
    steps = 0

    print("开始测试模型...")
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        env.render()

        if done or truncated:
            print(f"测试结束: 总奖励 = {total_reward:.2f}, 步数 = {steps}")
            break

    env.close()


def visualize_model(model_path, num_episodes=5, algorithm='ppo'):
    """
    加载并可视化已训练的模型

    Args:
        model_path: 模型文件路径（不含扩展名）
        num_episodes: 要运行的回合数
        algorithm: 算法类型 ('ppo' 或 'dqn')
    """
    import os

    # 检查模型文件是否存在
    if not os.path.exists(f"{model_path}.zip"):
        print(f"错误: 找不到模型文件 {model_path}.zip")
        print("请确保模型文件存在，或先运行训练脚本。")
        return

    # 加载模型
    print(f"正在加载模型: {model_path}.zip")
    try:
        if algorithm.lower() == 'ppo':
            model = PPO.load(model_path)
        elif algorithm.lower() == 'dqn':
            model = DQN.load(model_path)
        
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    print(f"模型加载成功！开始可视化 {num_episodes} 个回合...")
    print("按 ESC 键可以提前退出\n")

    # 创建带渲染的环境
    env = RacingEnv(render_mode='human')

    episode_rewards = []
    episode_steps = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        episode_done = False

        print(f"\n=== 回合 {episode + 1}/{num_episodes} ===")

        while not episode_done:
            # 检查是否要提前退出（通过 pygame 事件）
            import pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("\n用户提前退出")
                    episode_done = True
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        print("\n用户提前退出")
                        episode_done = True
                        break

            if episode_done:
                break

            # 预测动作
            action, _ = model.predict(obs, deterministic=True)

            # 执行动作
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            # 渲染
            env.render()

            # 检查是否结束
            if done or truncated:
                episode_done = True
                episode_rewards.append(total_reward)
                episode_steps.append(steps)
                print(f"回合 {episode + 1} 结束:")
                print(f"  总奖励: {total_reward:.2f}")
                print(f"  步数: {steps}")

        if episode < num_episodes - 1:
            # 短暂暂停，让用户看到结果
            import time
            time.sleep(1)

    env.close()

    # 打印统计信息
    print("\n" + "="*50)
    print("可视化统计:")
    print("="*50)
    print(f"总回合数: {len(episode_rewards)}")
    print(
        f"平均奖励: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"最高奖励: {np.max(episode_rewards):.2f}")
    print(f"最低奖励: {np.min(episode_rewards):.2f}")
    print(f"平均步数: {np.mean(episode_steps):.2f} ± {np.std(episode_steps):.2f}")
    print(f"最长步数: {np.max(episode_steps)}")
    print(f"最短步数: {np.min(episode_steps)}")
    print("="*50)


def random_policy_demo():
    """随机策略演示（用于测试环境）"""
    env = RacingEnv(render_mode='human')

    print("运行随机策略演示...")
    for episode in range(3):
        obs, info = env.reset()
        total_reward = 0
        steps = 0

        while True:
            # 随机动作
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            env.render()

            if done or truncated:
                print(
                    f"回合 {episode + 1} 结束: 总奖励 = {total_reward:.2f}, 步数 = {steps}")
                break

    env.close()


if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == 'ppo':
            train_with_ppo()
        elif command == 'random':
            random_policy_demo()
        elif command == 'visualize' or command == 'viz':
            # 可视化模式: python train_rl.py visualize [model_path] [algorithm] [num_episodes]
            if len(sys.argv) < 3:
                print(
                    "用法: python train_rl.py visualize <model_path> [algorithm] [num_episodes]")
                print("示例: python train_rl.py visualize ppo_racing_car ppo 5")
                sys.exit(1)

            model_path = sys.argv[2]
            algorithm = sys.argv[3] if len(sys.argv) > 3 else 'ppo'
            num_episodes = int(sys.argv[4]) if len(sys.argv) > 4 else 5

            visualize_model(model_path, num_episodes, algorithm)
        elif command == 'continue' or command == 'resume':
            # 继续训练模式: python train_rl.py continue <model_path> [algorithm] [total_timesteps] [save_freq]
            if len(sys.argv) < 3:
                print("用法: python train_rl.py continue <model_path> [algorithm] [total_timesteps] [save_freq]")
                print("示例: python train_rl.py continue ppo_racing_car ppo 100000 10000")
                print("示例: python train_rl.py continue ppo_racing_car ppo 50000")
                sys.exit(1)

            model_path = sys.argv[2]
            algorithm = sys.argv[3] if len(sys.argv) > 3 else 'ppo'
            total_timesteps = int(sys.argv[4]) if len(sys.argv) > 4 else 100000
            save_freq = int(sys.argv[5]) if len(sys.argv) > 5 else 10000

            continue_training(model_path, algorithm, total_timesteps, save_freq)
        else:
            print("用法:")
            print("  训练: python train_rl.py [ppo|dqn|random]")
            print(
                "  可视化: python train_rl.py visualize <model_path> [algorithm] [num_episodes]")
            print("  继续训练: python train_rl.py continue <model_path> [algorithm] [total_timesteps] [save_freq]")
            print("示例:")
            print("  python train_rl.py ppo")
            print("  python train_rl.py visualize ppo_racing_car ppo 5")
            print("  python train_rl.py continue ppo_racing_car ppo 100000 10000")
    else:
        # 默认运行随机策略演示
        print("未指定命令，运行随机策略演示...")
        print("用法:")
        print("  训练: python train_rl.py [ppo|dqn|random]")
        print(
            "  可视化: python train_rl.py visualize <model_path> [algorithm] [num_episodes]")
        print("  继续训练: python train_rl.py continue <model_path> [algorithm] [total_timesteps] [save_freq]")
        random_policy_demo()
