# train_rl.py
import os
import argparse
import numpy as np
import torch
from racingcar.racing_env import RacingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

# 全局配置
N_ENVS = 8  # CPU 核心数
N_STEPS = 128  # 运算步 需要同步调整
SEED = 42


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
            save_path = os.path.join(self.save_dir, f"{self.name_prefix}_{self.num_timesteps}")
            self.model.save(save_path)
            if self.verbose > 0:
                print(f"[Checkpoint] 已保存模型至 {save_path}.zip")
        return True


def get_vec_env(n_envs=N_ENVS, render_mode=None):
    """统一创建并行环境的辅助函数"""
    return make_vec_env(
        RacingEnv,
        n_envs=n_envs,
        env_kwargs={'render_mode': render_mode},
        vec_env_cls=SubprocVecEnv,
        seed=SEED
    )


def train(args):
    """从头开始训练"""
    print(f"启动训练: {N_ENVS} 核并行, 总步数 {args.timesteps}")
    env = get_vec_env()

    # 针对32核无显卡服务器优化后的参数
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=N_STEPS,  # 优化点: 32 envs * 512 steps = 16384 batch size
        batch_size=256,  # 优化点: CPU 推理 batch size
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        device="cpu",  # 显式指定 CPU
        tensorboard_log="./logs/",
    )

    checkpoint_callback = PeriodicCheckpointCallback(
        save_freq=args.save_freq,
        save_dir="./checkpoints/ppo",
        name_prefix="ppo_racing_car",
        verbose=1,
    )

    model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback)

    save_path = "ppo_racing_car_final"
    model.save(save_path)
    print(f"训练完成，模型已保存为 {save_path}.zip")
    env.close()


def continue_train(args):
    """继续训练"""
    model_path = args.model_path
    if not os.path.exists(f"{model_path}.zip") and not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        return

    print(f"加载模型: {model_path} ...")

    # 1. 创建环境 (必须在 load 之前)
    env = get_vec_env()

    # 2. 加载模型并注入环境
    try:
        model = PPO.load(model_path, env=env, device="cpu")
    except Exception as e:
        print(f"加载失败 (请检查 numpy 版本或文件路径): {e}")
        env.close()
        return

    print(f"当前已训练步数: {model.num_timesteps}")
    print(f"目标继续训练: {args.timesteps} 步")

    checkpoint_callback = PeriodicCheckpointCallback(
        save_freq=args.save_freq,
        save_dir="./checkpoints/ppo",
        name_prefix="ppo_racing_car",
        verbose=1,
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        reset_num_timesteps=False
    )

    model.save(model_path)
    print(f"模型已更新并保存至 {model_path}")
    env.close()


def visualize(args):
    """可视化模型表现 (单环境渲染)"""
    model_path = args.model_path
    if not os.path.exists(f"{model_path}.zip"):
        print(f"错误: 找不到模型 {model_path}")
        return

    print(f"加载模型进行可视化: {model_path}")
    try:
        model = PPO.load(model_path, device="cpu")
    except Exception as e:
        print(f"加载失败: {e}")
        return

    # 可视化只能用单进程 + render_mode='human'
    env = RacingEnv(render_mode='human')

    for episode in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        print(f"--- Episode {episode + 1}/{args.episodes} ---")
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            env.render()

            # 简单的退出检查
            import pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    print("用户退出")
                    env.close()
                    return

            if done or truncated:
                print(f"Episode 结束: Reward={total_reward:.2f}, Steps={steps}")
                done = True
                import time
                time.sleep(1)

    env.close()


def random_demo(args):
    """随机策略演示"""
    print("运行随机策略演示...")
    env = RacingEnv(render_mode='human')
    for _ in range(3):
        env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            _, _, done, truncated, _ = env.step(action)
            env.render()
            if done or truncated: break
    env.close()


def main():
    parser = argparse.ArgumentParser(description="赛车强化学习训练脚本 (PPO Only)")
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    subparsers.required = True

    # 1. Train 命令
    parser_train = subparsers.add_parser('train', help='从头开始训练')
    parser_train.add_argument('--timesteps', type=int, default=500000, help='训练总步数')
    parser_train.add_argument('--save_freq', type=int, default=20000, help='保存频率')
    parser_train.set_defaults(func=train)

    # 2. Continue 命令
    parser_cont = subparsers.add_parser('continue', help='加载模型继续训练')
    parser_cont.add_argument('model_path', type=str, help='模型路径 (不含 .zip)')
    parser_cont.add_argument('--timesteps', type=int, default=200000, help='继续训练步数')
    parser_cont.add_argument('--save_freq', type=int, default=20000, help='保存频率')
    parser_cont.set_defaults(func=continue_train)

    # 3. Visualize 命令
    parser_viz = subparsers.add_parser('viz', help='可视化模型')
    parser_viz.add_argument('model_path', type=str, help='模型路径')
    parser_viz.add_argument('--episodes', type=int, default=5, help='运行回合数')
    parser_viz.set_defaults(func=visualize)

    # 4. Random 命令
    parser_rand = subparsers.add_parser('random', help='随机策略测试')
    parser_rand.set_defaults(func=random_demo)

    # 解析参数并执行对应函数
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()