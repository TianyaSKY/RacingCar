# entities/obstacle.py
import random

from ..utils import draw_cube
from ..config import *
from OpenGL.GL import glColor4f

class Obstacle:
    def __init__(self, track_instance):
        self.track = track_instance
        self.respawn()

    def respawn(self):
        self.passed = False
        # 获取赛道上的点和法向量
        tx, tz, nx, nz = self.track.get_random_track_position()
        
        # 在道路宽度范围内随机偏移
        offset = random.uniform(-ROAD_WIDTH * 0.8, ROAD_WIDTH * 0.8)
        
        self.x = tx + nx * offset
        self.z = tz + nz * offset
        
        self.width = random.uniform(1.0, 1.5)
        self.depth = random.uniform(1.0, 1.5)
        self.height = random.uniform(1.5, 2.5)

    def draw(self):
        # 主体 - 橙色警示色（更符合真实路障）
        draw_cube(self.x, 0, self.z, self.width, self.height, self.depth, (1.0, 0.5, 0.0))
        # 顶部 - 亮黄色反光（增强立体感）
        draw_cube(self.x, self.height, self.z, self.width, self.height*0.1, self.depth, (1.0, 1.0, 0.0))
        
        # 添加黑色警示条纹（模拟真实路障）
        stripe_width = 0.1 * self.height
        for y_offset in [0.25, 0.55]:
            y_pos = y_offset * self.height
            draw_cube(self.x, y_pos, self.z, self.width, stripe_width, self.depth, (0.0, 0.0, 0.0))
        
        # 添加阴影效果（增强地面接触感）
        glColor4f(0.0, 0.0, 0.0, 0.3)
        draw_cube(self.x, -0.1, self.z, self.width*1.2, 0.05, self.depth*1.2, (0.0, 0.0, 0.0))

    def get_bounds(self):
        return (self.x - self.width/2, self.x + self.width/2,
                self.z - self.depth/2, self.z + self.depth/2)
