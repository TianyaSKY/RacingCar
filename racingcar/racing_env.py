# racing_env.py
"""
赛车强化学习环境
符合 OpenAI Gym/Gymnasium 接口
"""
import numpy as np
import math
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYMNASIUM = True
except ImportError:
    import gym
    from gym import spaces
    GYMNASIUM = False

from .config import *
from .utils import resize, draw_text_2d
from .entities.car import Car
from .entities.track import Track
from .entities.obstacle import Obstacle


class RacingEnv(gym.Env):
    """
    赛车强化学习环境
    
    状态空间（观察）:
    - 车辆位置 (x, z) - 归一化
    - 车辆角度 (归一化到 -1 到 1)
    - 车辆速度 (归一化)
    - 到赛道中心线的距离 (归一化)
    - 前方障碍物距离 (5个方向的射线检测，归一化)
    - 车辆到赛道中心线的方向 (归一化)
    
    动作空间:
    离散动作: 0-6
    0: 无动作
    1: 加速
    2: 减速
    3: 左转
    4: 右转
    5: 加速+左转
    6: 加速+右转
    
    奖励:
    - 每步生存: -0.1
    - 碰撞惩罚: -100
    - 出界惩罚: -50
    - 速度奖励: 速度 * 0.1
    - 保持在赛道上奖励: (1 - 距离/ROAD_WIDTH) * 0.5
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, render_mode=None):
        super(RacingEnv, self).__init__()
        
        self.render_mode = render_mode
        
        # 动作空间: 7个离散动作
        self.action_space = spaces.Discrete(7)
        
        # 状态空间: 
        # [x, z, angle, speed, track_dist, track_dir_x, track_dir_z, ray_0, ray_1, ray_2, ray_3, ray_4]
        # 12个特征
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(12,), 
            dtype=np.float32
        )
        
        # 初始化 Pygame 和 OpenGL（仅在需要渲染时）
        self._pygame_initialized = False
        self._init_pygame()
        
        # 游戏状态
        self.track = None
        self.car = None
        self.obstacles = None
        self.steps = 0
        self.max_steps = 2000  # 最大步数
        
        # 用于渲染
        self.clock = None
        if self.render_mode == 'human':
            self.clock = pygame.time.Clock()
    
    def _init_pygame(self):
        """初始化 Pygame 和 OpenGL（仅在需要时）"""
        if not self._pygame_initialized and self.render_mode == 'human':
            pygame.init()
            pygame.font.init()
            self.display = (DISPLAY_WIDTH, DISPLAY_HEIGHT)
            pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL | RESIZABLE)
            pygame.display.set_caption("Racing RL Environment")
            resize(DISPLAY_WIDTH, DISPLAY_HEIGHT)
            glEnable(GL_DEPTH_TEST)
            glShadeModel(GL_SMOOTH)
            self._pygame_initialized = True
    
    def _get_observation(self):
        """
        获取当前观察（状态）
        返回归一化的特征向量
        """
        # 1. 车辆位置（归一化到 -1 到 1，假设世界范围是 -400 到 400）
        norm_x = self.car.x / 400.0
        norm_z = self.car.z / 400.0
        
        # 2. 车辆角度（归一化到 -1 到 1）
        norm_angle = (self.car.angle % 360) / 180.0 - 1.0
        
        # 3. 车辆速度（归一化）
        norm_speed = self.car.current_speed / CAR_BASE_SPEED
        
        # 4. 到赛道中心线的距离（归一化）
        track_dist = self.track.get_closest_distance(self.car.x, self.car.z)
        norm_track_dist = track_dist / (ROAD_WIDTH * 2)
        
        # 5. 车辆到赛道中心线的方向（归一化）
        # 找到最近的赛道点
        closest_point = self._get_closest_track_point()
        if closest_point:
            dx = closest_point[0] - self.car.x
            dz = closest_point[1] - self.car.z
            dist = math.sqrt(dx*dx + dz*dz)
            if dist > 0:
                norm_track_dir_x = dx / (ROAD_WIDTH * 2)
                norm_track_dir_z = dz / (ROAD_WIDTH * 2)
            else:
                norm_track_dir_x = 0
                norm_track_dir_z = 0
        else:
            norm_track_dir_x = 0
            norm_track_dir_z = 0
        
        # 6. 射线检测（前方障碍物距离）
        rays = self._raycast_obstacles()
        
        # 组合观察向量
        obs = np.array([
            norm_x,
            norm_z,
            norm_angle,
            norm_speed,
            norm_track_dist,
            norm_track_dir_x,
            norm_track_dir_z,
            rays[0],  # 前方
            rays[1],  # 左前方
            rays[2],  # 右前方
            rays[3],  # 左侧
            rays[4],  # 右侧
        ], dtype=np.float32)
        
        return obs
    
    def _get_closest_track_point(self):
        """获取最近的赛道点"""
        min_dist_sq = float('inf')
        closest = None
        for px, pz in self.track.path_points:
            dx = self.car.x - px
            dz = self.car.z - pz
            d_sq = dx*dx + dz*dz
            if d_sq < min_dist_sq:
                min_dist_sq = d_sq
                closest = (px, pz)
        return closest
    
    def _raycast_obstacles(self, max_distance=50.0, num_rays=5):
        """
        射线检测障碍物
        返回归一化的距离（0-1，1表示没有障碍物）
        """
        rad = math.radians(self.car.angle)
        ray_angles = [0, -30, 30, -90, 90]  # 前方、左前方、右前方、左侧、右侧
        
        ray_distances = []
        for ray_angle in ray_angles:
            ray_rad = rad + math.radians(ray_angle)
            ray_dx = math.sin(ray_rad)
            ray_dz = math.cos(ray_rad)
            
            min_dist = max_distance
            
            # 检测与障碍物的碰撞
            for step in range(0, int(max_distance * 10), 1):
                t = step / 10.0
                ray_x = self.car.x + ray_dx * t
                ray_z = self.car.z + ray_dz * t
                
                # 检查是否与障碍物碰撞
                for obs in self.obstacles:
                    obs_bounds = obs.get_bounds()
                    if (obs_bounds[0] <= ray_x <= obs_bounds[1] and 
                        obs_bounds[2] <= ray_z <= obs_bounds[3]):
                        min_dist = t
                        break
                
                if min_dist < max_distance:
                    break
            
            # 归一化距离（0-1，1表示没有障碍物）
            norm_dist = min_dist / max_distance
            ray_distances.append(norm_dist)
        
        return ray_distances
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)
            import random
            random.seed(seed)
        
        # 重置游戏状态
        self.steps = 0
        
        # 生成新赛道
        self.track = Track()
        
        # 获取起点，放置车辆
        start_x, start_z, start_angle = self.track.get_start_position()
        self.car = Car(start_pos=(start_x, start_z, start_angle))
        
        # 生成障碍物
        self.obstacles = [Obstacle(self.track) for _ in range(OBSTACLE_COUNT)]
        
        # 获取初始观察
        observation = self._get_observation()
        info = {}
        
        if GYMNASIUM:
            return observation, info
        else:
            return observation
    
    def step(self, action):
        """执行一步"""
        self.steps += 1
        
        # 应用动作
        self.car.apply_action(action)
        
        # 检查碰撞和出界
        done = False
        reward = -0.5  # 每秒存活惩罚
        
        # 1. 碰撞检测（障碍物）
        car_bounds = self.car.get_bounds()
        for obs in self.obstacles:
            if self._check_aabb_collision(car_bounds, obs.get_bounds()):
                done = True
                reward = -100  # 碰撞惩罚
                break
        
        # 2. 出界检测
        if not done:
            dist = self.track.get_closest_distance(self.car.x, self.car.z)
            if dist > ROAD_WIDTH + OFF_ROAD_TOLERANCE:
                done = True
                reward = -50  # 出界惩罚
            else:
                # 速度奖励
                reward += self.car.current_speed * 0.8
                
                # 保持在赛道上的奖励
                track_reward = (1 - dist / ROAD_WIDTH) * 0.05
                reward += max(0, track_reward)
        
        # 3. 最大步数限制
        if self.steps >= self.max_steps:
            done = True
        
        # 获取新观察
        observation = self._get_observation()
        info = {'steps': self.steps}
        
        if GYMNASIUM:
            return observation, reward, done, False, info
        else:
            return observation, reward, done, info
    
    def _check_aabb_collision(self, b1, b2):
        """AABB 碰撞检测"""
        return (b1[0] < b2[1] and b1[1] > b2[0] and
                b1[2] < b2[3] and b1[3] > b2[2])
    
    def render(self):
        """渲染环境"""
        if self.render_mode != 'human':
            return None
        
        if not self._pygame_initialized:
            self._init_pygame()
        
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            elif event.type == VIDEORESIZE:
                resize(event.w, event.h)
        
        # 渲染
        glClearColor(*COLOR_SKY, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # 第三人称相机
        dist_behind = 15
        cam_height = 6
        rad = math.radians(self.car.angle)
        camera_x = self.car.x - math.sin(rad) * dist_behind
        camera_z = self.car.z - math.cos(rad) * dist_behind
        
        gluLookAt(camera_x, cam_height, camera_z, 
                  self.car.x, 0, self.car.z, 
                  0, 1, 0)
        
        # 绘制世界
        self.track.draw()
        self.car.draw()
        for obs in self.obstacles:
            obs.draw()
        
        # UI
        draw_text_2d(f"Steps: {self.steps}", 10, 10, 30, COLOR_TEXT_SCORE)
        draw_text_2d(f"Speed: {self.car.current_speed:.2f}", 10, 40, 30, COLOR_TEXT_SCORE)
        
        # 显示到赛道中心线的距离
        track_dist = self.track.get_closest_distance(self.car.x, self.car.z)
        draw_text_2d(f"Track Dist: {track_dist:.2f}", 10, 70, 30, COLOR_TEXT_SCORE)
        
        pygame.display.flip()
        if self.clock:
            self.clock.tick(60)
        
        return None
    
    def close(self):
        """关闭环境"""
        if self._pygame_initialized:
            pygame.quit()
            self._pygame_initialized = False

