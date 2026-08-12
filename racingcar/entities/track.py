# entities/track.py
import math
import random
from OpenGL.GL import *
from OpenGL.GLU import GLU_SMOOTH, gluCylinder, gluNewQuadric, gluQuadricNormals

from ..config import *

class Track:
    def __init__(self):
        self.path_points = [] # 赛道中心线的高密度点集
        self.left_edge = []
        self.right_edge = []
        self.generate_track()
        self._generate_trees()
        self._tree_quadric = gluNewQuadric()
        gluQuadricNormals(self._tree_quadric, GLU_SMOOTH)

    def _generate_trees(self):
        """在赛道两侧随机布置低多边形树木（固定种子，布局稳定）。"""
        rng = random.Random(20260812)
        self.trees = []
        for i in range(0, len(self.path_points), 3):
            p = self.path_points[i]
            # 复用赛道边缘方向作为外法线
            nx = (self.left_edge[i][0] - p[0]) / ROAD_WIDTH
            nz = (self.left_edge[i][1] - p[1]) / ROAD_WIDTH
            side = 1 if rng.random() < 0.5 else -1
            dist = ROAD_WIDTH + 9 + rng.random() * 16
            x = p[0] + side * nx * dist
            z = p[1] + side * nz * dist
            # 与已有树木保持间距，避免扎堆
            if all((x - tx) ** 2 + (z - tz) ** 2 > 49 for tx, tz, _ in self.trees):
                self.trees.append((x, z, 0.8 + rng.random() * 0.7))

    def _catmull_rom(self, p0, p1, p2, p3, t):
        """计算插值点，生成平滑曲线"""
        t2 = t * t
        t3 = t2 * t
        
        # Catmull-Rom 矩阵公式
        def solve(v0, v1, v2, v3):
            return 0.5 * ((2 * v1) +
                          (-v0 + v2) * t +
                          (2 * v0 - 5 * v1 + 4 * v2 - v3) * t2 +
                          (-v0 + 3 * v1 - 3 * v2 + v3) * t3)

        return (solve(p0[0], p1[0], p2[0], p3[0]), 
                solve(p0[1], p1[1], p2[1], p3[1]))

    def generate_track(self):
        # 1. 生成粗略的控制点 (极坐标 -> 直角坐标)
        control_points = []
        for i in range(TRACK_POINTS):
            angle = (2 * math.pi / TRACK_POINTS) * i
            # 随机半径，产生非完美圆形
            radius = TRACK_RADIUS + random.uniform(-TRACK_VARIANCE, TRACK_VARIANCE)
            x = math.cos(angle) * radius
            z = math.sin(angle) * radius
            control_points.append((x, z))

        # 2. 使用样条插值生成高密度路径
        self.path_points = []
        steps_per_segment = 10 # 每两个控制点之间插入多少个点
        
        for i in range(len(control_points)):
            p0 = control_points[(i - 1) % len(control_points)]
            p1 = control_points[i]
            p2 = control_points[(i + 1) % len(control_points)]
            p3 = control_points[(i + 2) % len(control_points)]

            for t in range(steps_per_segment):
                self.path_points.append(self._catmull_rom(p0, p1, p2, p3, t / steps_per_segment))

        # 3. 计算赛道边缘 (用于绘制)
        self.left_edge = []
        self.right_edge = []
        
        for i in range(len(self.path_points)):
            curr = self.path_points[i]
            # 计算切线向量
            next_p = self.path_points[(i + 1) % len(self.path_points)]
            prev_p = self.path_points[(i - 1) % len(self.path_points)]
            
            dx = next_p[0] - prev_p[0]
            dz = next_p[1] - prev_p[1]
            length = math.sqrt(dx*dx + dz*dz)
            if length == 0: length = 1
            
            # 法线向量 (垂直于切线)
            nx = -dz / length
            nz = dx / length
            
            # 扩展出左右边缘
            self.left_edge.append((curr[0] + nx * ROAD_WIDTH, curr[1] + nz * ROAD_WIDTH))
            self.right_edge.append((curr[0] - nx * ROAD_WIDTH, curr[1] - nz * ROAD_WIDTH))

    def get_start_position(self):
        """返回起点位置和初始朝向"""
        p0 = self.path_points[0]
        p1 = self.path_points[1]
        angle = math.atan2(p1[0] - p0[0], p1[1] - p0[1])
        angle_deg = math.degrees(angle)
        return p0[0], p0[1], angle_deg

    def get_closest_distance(self, x, z):
        """计算 (x, z) 到赛道中心线的最近距离 (用于碰撞检测)"""
        min_dist_sq = float('inf')
        # 简单优化：因为赛道是闭环，暴力遍历几百个点其实很快
        # 如果很卡，可以使用空间分区优化
        for px, pz in self.path_points:
            dx = x - px
            dz = z - pz
            d_sq = dx*dx + dz*dz
            if d_sq < min_dist_sq:
                min_dist_sq = d_sq
        return math.sqrt(min_dist_sq)

    def get_random_track_position(self):
        """返回赛道上的一个随机点和它的切线法向量 (用于放置障碍物)"""
        idx = random.randint(10, len(self.path_points) - 1) # 避开起点
        p = self.path_points[idx]
        
        # 计算局部法向量以便横向偏移
        next_p = self.path_points[(idx + 1) % len(self.path_points)]
        dx = next_p[0] - p[0]
        dz = next_p[1] - p[1]
        length = math.sqrt(dx*dx + dz*dz)
        nx, nz = -dz/length, dx/length
        
        return p[0], p[1], nx, nz

    def draw(self):
        self._draw_ground()
        self._draw_track_surface()
        self._draw_center_line()
        self._draw_trees()

    def _draw_ground(self):
        """棋盘格草地：深浅绿交替的网格在光照下呈现纹理感。"""
        glNormal3f(0.0, 1.0, 0.0)
        tile = 20.0
        half = 10
        for gx in range(-half, half):
            for gz in range(-half, half):
                if (gx + gz) % 2 == 0:
                    glColor3f(0.16, 0.38, 0.14)
                else:
                    glColor3f(0.12, 0.30, 0.11)
                glBegin(GL_QUADS)
                glVertex3f(gx * tile, -1.5, gz * tile)
                glVertex3f((gx + 1) * tile, -1.5, gz * tile)
                glVertex3f((gx + 1) * tile, -1.5, (gz + 1) * tile)
                glVertex3f(gx * tile, -1.5, (gz + 1) * tile)
                glEnd()

    def _draw_track_surface(self):
        """路肩红白棋盘 + 深灰沥青路面，法线朝上保证光照正确。"""
        num_points = len(self.path_points)
        glNormal3f(0.0, 1.0, 0.0)

        glBegin(GL_QUAD_STRIP)
        for i in range(num_points + 1):  # +1 是为了闭合环路
            idx = i % num_points
            if (i // 4) % 2 == 0:
                glColor3f(0.78, 0.10, 0.10)  # 鲜艳红色
            else:
                glColor3f(0.92, 0.92, 0.92)  # 亮白色
            lx, lz = self.left_edge[idx]
            rx, rz = self.right_edge[idx]
            glVertex3f(lx, -1.0, lz)
            glVertex3f(rx, -1.0, rz)
        glEnd()

        # 沥青路面（收缩露出路肩）
        glColor3f(0.24, 0.24, 0.25)
        glBegin(GL_QUAD_STRIP)
        for i in range(num_points + 1):
            idx = i % num_points
            cx, cz = self.path_points[idx]
            lx = cx + (self.left_edge[idx][0] - cx) * 0.9
            lz = cz + (self.left_edge[idx][1] - cz) * 0.9
            rx = cx + (self.right_edge[idx][0] - cx) * 0.9
            rz = cz + (self.right_edge[idx][1] - cz) * 0.9
            glVertex3f(lx, -0.9, lz)
            glVertex3f(rx, -0.9, rz)
        glEnd()

    def _draw_center_line(self):
        """黄色虚线中心线，比连续线更接近真实赛道。"""
        num_points = len(self.path_points)
        step = 3
        glColor3f(1.0, 0.92, 0.25)
        glNormal3f(0.0, 1.0, 0.0)
        glBegin(GL_QUADS)
        for i in range(0, num_points, step * 2):
            i0 = i % num_points
            i1 = (i + step) % num_points
            p0 = self.path_points[i0]
            p1 = self.path_points[i1]
            dx = p1[0] - p0[0]
            dz = p1[1] - p0[1]
            length = math.hypot(dx, dz)
            if length == 0:
                length = 1
            nx, nz = -dz / length, dx / length
            hw = 0.13
            glVertex3f(p0[0] + nx * hw, -0.85, p0[1] + nz * hw)
            glVertex3f(p0[0] - nx * hw, -0.85, p0[1] - nz * hw)
            glVertex3f(p1[0] - nx * hw, -0.85, p1[1] - nz * hw)
            glVertex3f(p1[0] + nx * hw, -0.85, p1[1] + nz * hw)
        glEnd()

    def _draw_trees(self):
        """树干圆柱 + 两层圆锥树冠的低多边形树木（竖直生长）。"""
        for x, z, scale in self.trees:
            glPushMatrix()
            glTranslatef(x, -1.5, z)
            glScalef(scale, scale, scale)
            # gluCylinder 沿 +Z 生成，旋转 +90° 使树轴对齐世界 +Y（竖直向上）
            glRotatef(90.0, 1.0, 0.0, 0.0)
            glColor3f(0.42, 0.30, 0.16)
            gluCylinder(self._tree_quadric, 0.28, 0.38, 2.2, 8, 1)
            glTranslatef(0.0, 0.0, 2.2)  # 旋转后本地 Z 即世界 Y
            glColor3f(0.10, 0.33, 0.12)
            gluCylinder(self._tree_quadric, 0.0, 1.7, 2.0, 8, 1)
            glTranslatef(0.0, 0.0, 1.5)
            glColor3f(0.13, 0.39, 0.15)
            gluCylinder(self._tree_quadric, 0.0, 1.2, 1.5, 8, 1)
            glPopMatrix()
