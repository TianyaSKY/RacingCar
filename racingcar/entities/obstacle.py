# entities/obstacle.py
import random

from ..utils import draw_cube
from ..config import *

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
        draw_cube(self.x, 0, self.z, self.width, self.height, self.depth, COLOR_OBSTACLE_BODY)
        draw_cube(self.x, self.height, self.z, self.width, self.height*0.1, self.depth, COLOR_OBSTACLE_TOP)

    def get_bounds(self):
        return (self.x - self.width/2, self.x + self.width/2,
                self.z - self.depth/2, self.z + self.depth/2)