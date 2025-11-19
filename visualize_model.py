# visualize_model.py
"""
独立的模型可视化脚本
可以更方便地加载和可视化已训练的模型
"""
import sys
import os
import numpy as np
from racingcar.racing_env import RacingEnv

try:
    from stable_baselines3 import PPO, DQN
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("错误: 需要安装 stable-baselines3")
    print("安装命令: pip install stable-baselines3")
    sys.exit(1)


def visualize_model(model_path, num_episodes=5, algorithm='auto'):
    """
    加载并可视化已训练的模型
    
    Args:
        model_path: 模型文件路径（不含扩展名）
        num_episodes: 要运行的回合数
        algorithm: 算法类型 ('ppo', 'dqn', 或 'auto' 自动检测)
    """
    # 检查模型文件是否存在
    model_file = f"{model_path}.zip"
    if not os.path.exists(model_file):
        print(f"错误: 找不到模型文件 {model_file}")
        print("请确保模型文件存在，或先运行训练脚本。")
        return
    
    # 加载模型
    print(f"正在加载模型: {model_file}")
    model = None
    
    if algorithm == 'auto':
        # 尝试自动检测
        try:
            model = PPO.load(model_path)
            print("✓ 自动检测为 PPO 模型")
        except:
            try:
                model = DQN.load(model_path)
                print("✓ 自动检测为 DQN 模型")
            except Exception as e:
                print(f"✗ 无法加载模型: {e}")
                return
    elif algorithm.lower() == 'ppo':
        try:
            model = PPO.load(model_path)
            print("✓ PPO 模型加载成功")
        except Exception as e:
            print(f"✗ 加载 PPO 模型失败: {e}")
            return
    elif algorithm.lower() == 'dqn':
        try:
            model = DQN.load(model_path)
            print("✓ DQN 模型加载成功")
        except Exception as e:
            print(f"✗ 加载 DQN 模型失败: {e}")
            return
    else:
        print(f"错误: 未知的算法类型 '{algorithm}'")
        return
    
    print(f"\n开始可视化 {num_episodes} 个回合...")
    print("提示: 按 ESC 键可以提前退出\n")
    
    # 创建带渲染的环境
    env = RacingEnv(render_mode='human')
    
    episode_rewards = []
    episode_steps = []
    action_names = ['无动作', '加速', '减速', '左转', '右转', '加速+左转', '加速+右转']
    
    try:
        for episode in range(num_episodes):
            obs, info = env.reset()
            total_reward = 0
            steps = 0
            episode_done = False
            action_counts = [0] * 7  # 统计每个动作的使用次数
            
            print(f"\n{'='*60}")
            print(f"回合 {episode + 1}/{num_episodes}")
            print(f"{'='*60}")
            
            while not episode_done:
                # 检查是否要提前退出
                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("\n用户提前退出（关闭窗口）")
                        episode_done = True
                        break
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            print("\n用户提前退出（按 ESC）")
                            episode_done = True
                            break
                
                if episode_done:
                    break
                
                # 预测动作
                action, _ = model.predict(obs, deterministic=True)
                action_counts[action] += 1
                
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
                    
                    # 打印回合结果
                    print(f"\n回合 {episode + 1} 完成:")
                    print(f"  总奖励: {total_reward:.2f}")
                    print(f"  步数: {steps}")
                    if done:
                        if total_reward < -50:
                            print(f"  结束原因: 碰撞障碍物")
                        elif total_reward < 0:
                            print(f"  结束原因: 偏离赛道")
                        else:
                            print(f"  结束原因: 达到最大步数")
                    
                    # 打印动作统计
                    print(f"  动作使用统计:")
                    for i, count in enumerate(action_counts):
                        if count > 0:
                            percentage = count / steps * 100
                            print(f"    {action_names[i]}: {count} 次 ({percentage:.1f}%)")
            
            if episode < num_episodes - 1:
                # 短暂暂停
                import time
                time.sleep(1.5)
    
    except KeyboardInterrupt:
        print("\n\n用户中断（Ctrl+C）")
    finally:
        env.close()
    
    # 打印总体统计信息
    if episode_rewards:
        print("\n" + "="*60)
        print("总体统计:")
        print("="*60)
        print(f"总回合数: {len(episode_rewards)}")
        print(f"\n奖励统计:")
        print(f"  平均奖励: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"  最高奖励: {np.max(episode_rewards):.2f}")
        print(f"  最低奖励: {np.min(episode_rewards):.2f}")
        print(f"\n步数统计:")
        print(f"  平均步数: {np.mean(episode_steps):.2f} ± {np.std(episode_steps):.2f}")
        print(f"  最长步数: {np.max(episode_steps)}")
        print(f"  最短步数: {np.min(episode_steps)}")
        print("="*60)


def list_models():
    """列出当前目录下的所有模型文件"""
    models = []
    for file in os.listdir('.'):
        if file.endswith('.zip'):
            model_name = file[:-4]  # 去掉 .zip 扩展名
            models.append(model_name)
    
    if models:
        print("找到以下模型文件:")
        for i, model in enumerate(models, 1):
            print(f"  {i}. {model}")
        return models
    else:
        print("当前目录下没有找到模型文件（.zip）")
        return []


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("模型可视化工具")
        print("="*60)
        print("用法:")
        print("  python visualize_model.py <model_path> [algorithm] [num_episodes]")
        print("\n参数:")
        print("  model_path    : 模型文件路径（不含 .zip 扩展名）")
        print("  algorithm     : 算法类型 ('ppo', 'dqn', 或 'auto' 自动检测，默认: auto)")
        print("  num_episodes  : 要运行的回合数（默认: 5）")
        print("\n示例:")
        print("  python visualize_model.py ppo_racing_car")
        print("  python visualize_model.py ppo_racing_car ppo 10")
        print("  python visualize_model.py dqn_racing_car dqn 3")
        print("\n可用模型:")
        list_models()
        sys.exit(0)
    
    model_path = sys.argv[1]
    algorithm = sys.argv[2] if len(sys.argv) > 2 else 'auto'
    num_episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    
    visualize_model(model_path, num_episodes, algorithm)

