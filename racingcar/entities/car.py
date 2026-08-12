# entities/car.py
from OpenGL.GL import *
from OpenGL.GLU import GLU_SMOOTH, gluCylinder, gluDisk, gluNewQuadric, gluQuadricNormals
import pygame
import math

from ..utils import draw_cube
from ..config import *

class Car:
    def __init__(self, start_pos=(0, 0, 0)):
        self.start_pos = start_pos
        self.reset()
        self._wheel_quadric = None

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
        """绘制一辆低趴、带空气动力套件的双座赛车。

        车辆的局部前方与运动方向一致（+Z）；碰撞尺寸保持不变。
        """
        glPushMatrix()
        # 轮胎半径为 0.34，根节点使其底部恰好落在路面 y = -0.9。
        glTranslatef(self.x, -0.16, self.z)
        glRotatef(self.angle, 0, 1, 0)

        self._draw_body_local()
        self._draw_cockpit_local()
        self._draw_aero_local()
        self._draw_lights_local()
        self._draw_wheels_local()

        glPopMatrix()

    def _draw_body_local(self):
        body_color = (0.05, 0.16, 0.43)
        accent_color = (0.0, 0.62, 0.98)
        dark_trim = (0.018, 0.025, 0.045)

        # 低矮而逐渐收窄的单体壳，比纯立方体车身更接近跑车比例。
        self._draw_tapered_prism(
            rear_z=-1.12, front_z=1.20,
            rear_width=1.42, front_width=1.20,
            rear_bottom=-0.26, front_bottom=-0.24,
            rear_top=0.13, front_top=-0.03,
            color=body_color,
        )
        draw_cube(0, -0.02, -0.88, 1.38, 0.27, 0.58, body_color, shininess=72.0)

        # 车身肩线、底部分流器和赛车涂装条带。
        draw_cube(0, -0.31, 1.20, 1.50, 0.08, 0.36, dark_trim)
        draw_cube(0, -0.30, -1.16, 1.48, 0.08, 0.30, dark_trim)
        draw_cube(0, 0.09, 0.92, 0.52, 0.05, 0.78, accent_color)
        draw_cube(-0.58, -0.03, 0.15, 0.07, 0.22, 1.62, accent_color)
        draw_cube(0.58, -0.03, 0.15, 0.07, 0.22, 1.62, accent_color)

    def _draw_tapered_prism(
            self, rear_z, front_z, rear_width, front_width,
            rear_bottom, front_bottom, rear_top, front_top, color):
        """绘制前低后高、前窄后宽的车身单体壳。"""
        rear_half, front_half = rear_width / 2, front_width / 2
        glColor3f(*color)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (0.52, 0.60, 0.72, 1.0))
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 76.0)
        glBegin(GL_QUADS)

        glNormal3f(0.0, 0.0, 1.0)
        glVertex3f(-front_half, front_bottom, front_z)
        glVertex3f(front_half, front_bottom, front_z)
        glVertex3f(front_half, front_top, front_z)
        glVertex3f(-front_half, front_top, front_z)

        glNormal3f(0.0, 0.0, -1.0)
        glVertex3f(-rear_half, rear_bottom, rear_z)
        glVertex3f(-rear_half, rear_top, rear_z)
        glVertex3f(rear_half, rear_top, rear_z)
        glVertex3f(rear_half, rear_bottom, rear_z)

        glNormal3f(0.0, 0.98, 0.20)
        glVertex3f(-rear_half, rear_top, rear_z)
        glVertex3f(-front_half, front_top, front_z)
        glVertex3f(front_half, front_top, front_z)
        glVertex3f(rear_half, rear_top, rear_z)

        glNormal3f(0.0, -1.0, 0.0)
        glVertex3f(-rear_half, rear_bottom, rear_z)
        glVertex3f(rear_half, rear_bottom, rear_z)
        glVertex3f(front_half, front_bottom, front_z)
        glVertex3f(-front_half, front_bottom, front_z)

        glNormal3f(1.0, 0.08, 0.0)
        glVertex3f(rear_half, rear_bottom, rear_z)
        glVertex3f(rear_half, rear_top, rear_z)
        glVertex3f(front_half, front_top, front_z)
        glVertex3f(front_half, front_bottom, front_z)

        glNormal3f(-1.0, 0.08, 0.0)
        glVertex3f(-rear_half, rear_bottom, rear_z)
        glVertex3f(-front_half, front_bottom, front_z)
        glVertex3f(-front_half, front_top, front_z)
        glVertex3f(-rear_half, rear_top, rear_z)
        glEnd()

    def _draw_cockpit_local(self):
        glass_color = (0.035, 0.11, 0.20)
        glass_highlight = (0.18, 0.72, 0.95)
        trim_color = (0.015, 0.02, 0.035)

        # 以三片低矮玻璃构成座舱，避免单个方块带来的“盒子感”。
        draw_cube(0, 0.24, -0.18, 1.08, 0.46, 0.92, glass_color)
        draw_cube(0, 0.40, -0.32, 0.92, 0.15, 0.62, glass_color)
        draw_cube(0, 0.34, 0.30, 0.98, 0.18, 0.22, glass_color)
        draw_cube(0, 0.52, -0.17, 0.04, 0.05, 0.85, glass_highlight)
        draw_cube(-0.56, 0.29, -0.16, 0.06, 0.31, 0.86, trim_color)
        draw_cube(0.56, 0.29, -0.16, 0.06, 0.31, 0.86, trim_color)

        # 后视镜及其细杆。
        for side in (-1, 1):
            draw_cube(side * 0.69, 0.25, 0.22, 0.10, 0.07, 0.22, trim_color)
            draw_cube(side * 0.74, 0.27, 0.22, 0.16, 0.09, 0.13, glass_color)

    def _draw_aero_local(self):
        carbon = (0.012, 0.016, 0.026)
        accent = (0.0, 0.48, 0.82)

        # 前唇和双层尾翼为车身提供清晰的竞技感。
        draw_cube(0, -0.20, 1.34, 1.56, 0.08, 0.18, carbon)
        draw_cube(-0.67, -0.15, 1.24, 0.10, 0.15, 0.35, carbon)
        draw_cube(0.67, -0.15, 1.24, 0.10, 0.15, 0.35, carbon)
        draw_cube(0, 0.55, -1.20, 1.48, 0.09, 0.20, carbon)
        draw_cube(0, 0.43, -1.12, 1.22, 0.06, 0.14, accent)
        draw_cube(-0.52, 0.17, -1.16, 0.08, 0.40, 0.10, carbon)
        draw_cube(0.52, 0.17, -1.16, 0.08, 0.40, 0.10, carbon)

    def _draw_lights_local(self):
        lens = (0.82, 0.94, 1.0)
        glow = (0.18, 0.72, 1.0)
        tail = (1.0, 0.045, 0.035)
        housing = (0.015, 0.02, 0.03)

        # 前灯置于 +Z 一端；后方镜头中同样可辨认细长尾灯。
        for side in (-1, 1):
            draw_cube(side * 0.43, 0.05, 1.29, 0.35, 0.12, 0.06, housing)
            draw_cube(side * 0.43, 0.08, 1.33, 0.28, 0.08, 0.04, lens)
            draw_cube(side * 0.43, 0.11, 1.35, 0.16, 0.03, 0.02, glow)
            draw_cube(side * 0.43, 0.08, -1.20, 0.32, 0.10, 0.05, tail)

        draw_cube(0, 0.08, -1.21, 0.32, 0.08, 0.05, tail)
        draw_cube(0, -0.04, -1.22, 0.26, 0.10, 0.05, housing)

    def _draw_wheels_local(self):
        radius, width = 0.34, 0.26
        tire_color = (0.012, 0.014, 0.019)
        rim_color = (0.48, 0.58, 0.66)
        hub_color = (0.02, 0.12, 0.22)
        caliper_color = (1.0, 0.20, 0.04)
        offsets = [
            (-self.width / 2 - width / 2 + 0.04, 0.82),
            (self.width / 2 + width / 2 - 0.04, 0.82),
            (-self.width / 2 - width / 2 + 0.04, -0.82),
            (self.width / 2 + width / 2 - 0.04, -0.82),
        ]

        if self._wheel_quadric is None:
            self._wheel_quadric = gluNewQuadric()
            gluQuadricNormals(self._wheel_quadric, GLU_SMOOTH)

        quadric = self._wheel_quadric
        for x, z in offsets:
            glPushMatrix()
            glTranslatef(x, -0.40, z)
            glRotatef(90 if x > 0 else -90, 0, 1, 0)

            # 圆柱轮胎与轮圈沿 X 轴排列，摆脱原先方形轮胎。
            glColor3f(*tire_color)
            gluCylinder(quadric, radius, radius, width, 20, 2)
            glColor3f(*rim_color)
            gluDisk(quadric, 0.0, radius * 0.60, 18, 1)
            glTranslatef(0, 0, width + 0.002)
            glColor3f(*rim_color)
            gluDisk(quadric, 0.0, radius * 0.60, 18, 1)
            glColor3f(*hub_color)
            gluDisk(quadric, 0.0, radius * 0.20, 14, 1)
            glPopMatrix()

            # 外露红色制动卡钳，使侧视轮组更有层次。
            draw_cube(x, -0.46, z, width + 0.02, 0.13, 0.20, caliper_color)

    def get_bounds(self):
        # 简单的 AABB，这里不处理旋转后的精确碰撞，否则太复杂
        # 只要车不是特别长条形，这通常足够了
        margin = 0.5
        return (self.x - self.width/2 + margin, self.x + self.width/2 - margin,
                self.z - self.depth/2 + margin, self.z + self.depth/2 - margin)
