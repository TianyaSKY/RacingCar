# entities/environment.py
from OpenGL.GL import *

from ..config import *

class Environment:
    def draw(self, player_z):
        self._draw_ground(player_z)
        self._draw_road(player_z)
        self._draw_road_markings(player_z)

    def _draw_ground(self, player_z):
        strip_size = 50.0
        num_strips = int(250 / strip_size) + 2
        
        for i in range(-num_strips, num_strips):
            color = COLOR_GROUND_LIGHT if i % 2 == 0 else COLOR_GROUND_DARK
            glColor3f(*color)
            start_z = int((player_z - 100) / strip_size) * strip_size + i * strip_size
            
            glBegin(GL_QUADS)
            glVertex3f(-100, -1, start_z + strip_size)
            glVertex3f(100, -1, start_z + strip_size)
            glVertex3f(100, -1, start_z)
            glVertex3f(-100, -1, start_z)
            glEnd()

    def _draw_road(self, player_z):
        glColor3f(*COLOR_ROAD)
        glBegin(GL_QUADS)
        
        w = ROAD_WIDTH 
        glVertex3f(-w, -0.9, player_z + 50); glVertex3f(w, -0.9, player_z + 50)
        glVertex3f(w, -0.9, player_z - 200); glVertex3f(-w, -0.9, player_z - 200)
        glEnd()

    def _draw_road_markings(self, player_z):
        line_w, line_l, line_g = 0.2, 3, 4
        total_unit = line_l + line_g
        num_lines = int(250 / total_unit) + 2
        
        glColor3f(*COLOR_ROAD_MARKING)
        glBegin(GL_QUADS)
        for i in range(-num_lines, num_lines):
            start_z = int((player_z - 100) / total_unit) * total_unit + i * total_unit
            glVertex3f(-line_w/2, -0.89, start_z + line_l); glVertex3f(line_w/2, -0.89, start_z + line_l)
            glVertex3f(line_w/2, -0.89, start_z); glVertex3f(-line_w/2, -0.89, start_z)
        glEnd()