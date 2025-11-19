# entities/track.py
import math
import random
from OpenGL.GL import *
from config import *

class Track:
    def __init__(self):
        self.path_points = [] # 赛道中心线的高密度点集
        self.left_edge = []
        self.right_edge = []
        self.generate_track()

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
        # 绘制无限大地板
        glColor3f(*COLOR_GROUND)
        glBegin(GL_QUADS)
        ground_size = 400
        glVertex3f(-ground_size, -1.5, -ground_size)
        glVertex3f(ground_size, -1.5, -ground_size)
        glVertex3f(ground_size, -1.5, ground_size)
        glVertex3f(-ground_size, -1.5, ground_size)
        glEnd()

        # 绘制赛道
        # 使用 Triangle Strip 或 Quad Strip 连接边缘点
        num_points = len(self.path_points)
        
        glBegin(GL_QUAD_STRIP)
        for i in range(num_points + 1): # +1 是为了闭合环路
            idx = i % num_points
            
            # 路肩变色效果
            if (i // 4) % 2 == 0:
                glColor3f(*COLOR_KERB_RED) 
            else:
                glColor3f(*COLOR_KERB_WHITE)
            
            # 稍微画宽一点作为路肩
            lx, lz = self.left_edge[idx]
            rx, rz = self.right_edge[idx]
            
            glVertex3f(lx, -1.0, lz)
            glVertex3f(rx, -1.0, rz)
            
        glEnd()

        # 绘制路面 (覆盖在路肩上面一点点)
        glColor3f(*COLOR_ROAD)
        glBegin(GL_QUAD_STRIP)
        for i in range(num_points + 1):
            idx = i % num_points
            # 收缩一点点，露出路肩
            lx, lz = self.left_edge[idx]
            rx, rz = self.right_edge[idx]
            
            # 简单的向量插值收缩
            cx, cz = self.path_points[idx]
            lx = cx + (lx - cx) * 0.9
            lz = cz + (lz - cz) * 0.9
            rx = cx + (rx - cx) * 0.9
            rz = cz + (rz - cz) * 0.9

            glVertex3f(lx, -0.9, lz)
            glVertex3f(rx, -0.9, rz)
        glEnd()