# entities/car.py
from OpenGL.GL import glPopMatrix, glPushMatrix, glRotatef, glTranslatef
import pygame
import math

from ..utils import draw_cube
from ..config import *

class Car:
    def __init__(self, start_pos=(0, 0, 0)):
        self.start_pos = start_pos
        self.reset()

    def reset(self):
        # 解包初始位置: x, z, angle(degrees)
        self.x, self.z, self.angle = self.start_pos
        self.current_speed = 0
        self.target_speed = 0
        
        self.width = 1.5
        self.height = 0.8
        self.depth = 2.5

    def update(self):
        keys = pygame.key.get_pressed()
        
        # --- 速度控制 (W / S) ---
        if keys[pygame.K_w]:
            self.target_speed = CAR_BASE_SPEED
        elif keys[pygame.K_s]:
            self.target_speed = -CAR_BASE_SPEED / 2
        else:
            self.target_speed = 0

        # 平滑加速
        if self.current_speed < self.target_speed:
            self.current_speed = min(self.target_speed, self.current_speed + ACCELERATION)
        elif self.current_speed > self.target_speed:
            self.current_speed = max(self.target_speed, self.current_speed - ACCELERATION)
        else:
            self.current_speed *= FRICTION

        # --- 转向控制 (Q / E) ---
        # 只有车在动的时候才能转向 (更真实)
        if abs(self.current_speed) > 0.05:
            direction = 1 if self.current_speed > 0 else -1
            if keys[pygame.K_q]:
                self.angle += TURN_SPEED * direction
            if keys[pygame.K_e]:
                self.angle -= TURN_SPEED * direction

        # --- 物理运动计算 ---
        # 将角度转换为弧度
        rad = math.radians(self.angle)
        # OpenGL 坐标系: X是右, -Z是前. 
        # sin/cos 的计算取决于你的 0度 定义。
        # 这里假设 0度 是沿着 Z轴正方向，所以前进是 sin(angle), cos(angle)
        # 经过调整适配 OpenGL 的 gluLookAt:
        self.x += math.sin(rad) * self.current_speed
        self.z += math.cos(rad) * self.current_speed

    def apply_action(self, action):
        """
        应用动作（用于强化学习）
        action: 整数，0-6
        0: 无动作
        1: 加速
        2: 减速
        3: 左转
        4: 右转
        5: 加速+左转
        6: 加速+右转
        """
        # 速度控制
        if action in [1, 5, 6]:  # 加速
            self.target_speed = CAR_BASE_SPEED
        elif action == 2:  # 减速
            self.target_speed = -CAR_BASE_SPEED / 2
        else:
            self.target_speed = 0

        # 平滑加速
        if self.current_speed < self.target_speed:
            self.current_speed = min(self.target_speed, self.current_speed + ACCELERATION)
        elif self.current_speed > self.target_speed:
            self.current_speed = max(self.target_speed, self.current_speed - ACCELERATION)
        else:
            self.current_speed *= FRICTION

        # 转向控制
        if abs(self.current_speed) > 0.05:
            direction = 1 if self.current_speed > 0 else -1
            if action in [3, 5]:  # 左转
                self.angle += TURN_SPEED * direction
            if action in [4, 6]:  # 右转
                self.angle -= TURN_SPEED * direction

        # 物理运动计算
        rad = math.radians(self.angle)
        self.x += math.sin(rad) * self.current_speed
        self.z += math.cos(rad) * self.current_speed

    def draw(self):
        glPushMatrix()
        glTranslatef(self.x, 0.4, self.z)
        glRotatef(self.angle, 0, 1, 0) # 旋转车身
        
        # 注意：这里不用再传 x, y, z 给 draw_cube，因为我们已经 translate 了
        # 只需要画在局部坐标 (0,0,0) 即可
        
        # 车身
        draw_cube(0, 0, 0, self.width, self.height, self.depth, COLOR_CAR_BODY)
        # 驾驶舱
        draw_cube(0, self.height/2 + 0.25 - 0.1, -0.2, 1.2, 0.5, 1.0, COLOR_CAR_WINDOW)
        
        # 车轮绘制 (局部坐标)
        self._draw_wheels_local()
        
        glPopMatrix()

    def _draw_wheels_local(self):
        r, t = 0.3, 0.2
        offsets = [
            (-self.width/2 - t/2 + 0.05, self.depth/2 - r),
            ( self.width/2 + t/2 - 0.05, self.depth/2 - r),
            (-self.width/2 - t/2 + 0.05, -self.depth/2 + r),
            ( self.width/2 + t/2 - 0.05, -self.depth/2 + r)
        ]
        for ox, oz in offsets:
            # y轴偏移是 r-0.4 (因为车身中心在0.4)
            draw_cube(ox, r - 0.4, oz, t, r*2, r*2, COLOR_CAR_WHEEL)

    def get_bounds(self):
        # 简单的 AABB，这里不处理旋转后的精确碰撞，否则太复杂
        # 只要车不是特别长条形，这通常足够了
        margin = 0.5
        return (self.x - self.width/2 + margin, self.x + self.width/2 - margin,
                self.z - self.depth/2 + margin, self.z + self.depth/2 - margin)