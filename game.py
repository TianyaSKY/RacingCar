# game.py
import pygame
import sys
import math
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

from config import *
from utils import resize, draw_text_2d
from entities.car import Car
from entities.track import Track
from entities.obstacle import Obstacle

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

        self.reset_game()
        
        self.running = True
        self.clock = pygame.time.Clock()

    def reset_game(self):
        self.game_over = False
        self.score = 0
        self.start_time = pygame.time.get_ticks()
        self.game_over_reason = ""
        
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

        # 1. 碰撞检测 (障碍物)
        car_bounds = self.car.get_bounds()
        for obs in self.obstacles:
            if self.check_aabb_collision(car_bounds, obs.get_bounds()):
                self.game_over = True
                self.game_over_reason = "Crashed into Obstacle"

        # 2. 出界检测 (核心逻辑修改)
        # 计算车距离赛道中心线的距离
        dist = self.track.get_closest_distance(self.car.x, self.car.z)
        if dist > ROAD_WIDTH + OFF_ROAD_TOLERANCE:
            self.game_over = True
            self.game_over_reason = "Off Track (You Fell Off!)"

        # 3. 简单的分数计算 (生存时间)
        self.score = (pygame.time.get_ticks() - self.start_time) // 100

    def check_aabb_collision(self, b1, b2):
        return (b1[0] < b2[1] and b1[1] > b2[0] and
                b1[2] < b2[3] and b1[3] > b2[2])

    def render(self):
        glClearColor(*COLOR_SKY, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # --- 第三人称旋转相机 ---
        # 相机需要位于车屁股后面，这需要用到三角函数计算位置
        dist_behind = 15
        cam_height = 6
        
        # 将角度转为弧度
        rad = math.radians(self.car.angle)
        
        # 关键：相机位置 = 车位置 - 车的朝向向量 * 距离
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
        draw_text_2d(f"Time: {self.score}", 10, 10, 30, COLOR_TEXT_SCORE)
        
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