# game.py
import pygame
import sys
import math
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

from .config import *
from .utils import resize, draw_text_2d
from .entities.car import Car
from .entities.track import Track
from .entities.obstacle import Obstacle

class Game:
    def __init__(self):
        pygame.init()
        pygame.font.init()
        self.display = (DISPLAY_WIDTH, DISPLAY_HEIGHT)
        pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL | RESIZABLE)
        pygame.display.set_caption("PyOpenGL Drift Racer")
        
        resize(DISPLAY_WIDTH, DISPLAY_HEIGHT)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)
        glEnable(GL_NORMALIZE)

        # 低密度暖雾把远处赛道融入天空，同时避免近处赛车发灰。
        glEnable(GL_FOG)
        fog_color = (0.30, 0.49, 0.68, 1.0)
        glFogfv(GL_FOG_COLOR, fog_color)
        glFogf(GL_FOG_DENSITY, 0.008)
        glFogi(GL_FOG_MODE, GL_EXP2)
        glHint(GL_FOG_HINT, GL_NICEST)

        # 日光主光 + 冷色补光：高光勾勒漆面，阴影面仍保留几何细节。
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        glLightfv(GL_LIGHT0, GL_POSITION, (-18.0, 22.0, 14.0, 1.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.16, 0.18, 0.22, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0, 0.88, 0.68, 1.0))
        glLightfv(GL_LIGHT0, GL_SPECULAR, (1.0, 0.96, 0.82, 1.0))
        glLightfv(GL_LIGHT1, GL_POSITION, (14.0, 8.0, -18.0, 1.0))
        glLightfv(GL_LIGHT1, GL_DIFFUSE, (0.22, 0.42, 0.72, 1.0))
        glLightfv(GL_LIGHT1, GL_SPECULAR, (0.18, 0.35, 0.65, 1.0))

        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        self.reset_game()
        
        self.running = True
        self.clock = pygame.time.Clock()

    def reset_game(self):
        self.game_over = False
        self.score = 0
        self.start_time = pygame.time.get_ticks()
        self.game_over_reason = ""
        self.current_reward = 0.0  # 当前奖励值
        self.total_reward = 0.0  # 累计奖励值
        
        # 1. 生成赛道
        self.track = Track()
        
        # 2. 获取赛道起点，放置车辆
        start_x, start_z, start_angle = self.track.get_start_position()
        self.car = Car(start_pos=(start_x, start_z, start_angle))
        
        # 3. 生成障碍物
        self.obstacles = [Obstacle(self.track) for _ in range(OBSTACLE_COUNT)]

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == VIDEORESIZE:
                resize(event.w, event.h)
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self.running = False
                if self.game_over and event.key == K_r:
                    self.reset_game()

    def update(self):
        if self.game_over:
            return

        self.car.update()

        # 计算奖励（与 racing_env.py 中的逻辑一致）
        reward = REWARD_STEP_PENALTY  # 每步存活惩罚

        # 1. 碰撞检测 (障碍物)
        car_bounds = self.car.get_bounds()
        collision_detected = False
        for obs in self.obstacles:
            if self.check_aabb_collision(car_bounds, obs.get_bounds()):
                self.game_over = True
                self.game_over_reason = "Crashed into Obstacle"
                reward = REWARD_COLLISION_PENALTY  # 碰撞惩罚
                collision_detected = True
                break

        # 2. 出界检测 (核心逻辑修改)
        # 计算车距离赛道中心线的距离
        if not collision_detected:
            dist = self.track.get_closest_distance(self.car.x, self.car.z)
            if dist > ROAD_WIDTH + OFF_ROAD_TOLERANCE:
                self.game_over = True
                self.game_over_reason = "Off Track (You Fell Off!)"
                reward = REWARD_OUT_OF_BOUNDS_PENALTY  # 出界惩罚
            else:
                # 速度奖励
                reward += self.car.current_speed * REWARD_SPEED_REWARD

                # 保持在赛道上的奖励
                track_reward = (1 - dist / ROAD_WIDTH) * REWARD_TRACK_CENTER_REWARD
                reward += max(0, track_reward)

        # 更新奖励值（用于显示）
        self.current_reward = reward
        self.total_reward += reward

        # 3. 简单的分数计算 (生存时间)
        self.score = (pygame.time.get_ticks() - self.start_time) // 100

    def check_aabb_collision(self, b1, b2):
        return (b1[0] < b2[1] and b1[1] > b2[0] and
                b1[2] < b2[3] and b1[3] > b2[2])

    def render(self):
        glClearColor(0.30, 0.49, 0.68, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # 相机略偏左且注视前方，能同时读到车顶、轮组和前部空气动力套件。
        dist_behind = 12.5
        cam_height = 4.4
        side_offset = 2.4
        look_ahead = 1.8
        rad = math.radians(self.car.angle)
        forward_x, forward_z = math.sin(rad), math.cos(rad)
        right_x, right_z = math.cos(rad), -math.sin(rad)
        camera_x = (
            self.car.x - forward_x * dist_behind + right_x * side_offset)
        camera_z = (
            self.car.z - forward_z * dist_behind + right_z * side_offset)

        gluLookAt(
            camera_x, cam_height, camera_z,
            self.car.x + forward_x * look_ahead, -0.18,
            self.car.z + forward_z * look_ahead,
            0, 1, 0,
        )

        # 绘制世界
        self.track.draw()
        self.car.draw()
        for obs in self.obstacles:
            obs.draw()

        # UI
        draw_text_2d(f"Time: {self.score}", 10, 10, 30, COLOR_TEXT_SCORE)
        draw_text_2d(f"Speed: {self.car.current_speed:.2f}",
                     10, 40, 30, COLOR_TEXT_SCORE)
        
        # 显示到赛道中心线的距离
        track_dist = self.track.get_closest_distance(self.car.x, self.car.z)
        draw_text_2d(f"Track Dist: {track_dist:.2f}",
                     10, 70, 30, COLOR_TEXT_SCORE)
        
        # 显示当前奖励值和累计奖励
        reward_color = COLOR_TEXT_SCORE if self.current_reward >= 0 else COLOR_TEXT_GAME_OVER
        draw_text_2d(f"Reward: {self.current_reward:.2f}",
                     10, 100, 30, reward_color)
        draw_text_2d(f"Total Reward: {self.total_reward:.2f}",
                     10, 130, 30, COLOR_TEXT_SCORE)
        
        if self.game_over:
            cx, cy = DISPLAY_WIDTH // 2, DISPLAY_HEIGHT // 2
            draw_text_2d("GAME OVER!", cx - 100, cy - 50, 60, COLOR_TEXT_GAME_OVER)
            draw_text_2d(f"{self.game_over_reason}", cx - 150, cy + 10, 30, COLOR_TEXT_INFO)
            draw_text_2d("Press 'R' to New Track", cx - 120, cy + 50, 30, COLOR_TEXT_INFO)

        pygame.display.flip()

    def run(self):
        while self.running:
            self.handle_input()
            self.update()
            self.render()
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()
