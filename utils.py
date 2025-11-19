# utils.py
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
from config import *

def resize(width, height):
    """处理窗口大小调整"""
    if height == 0: height = 1
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(FOV, (width / height), 0.1, 200.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def draw_cube(x, y, z, width, height, depth, color):
    """通用立方体绘制函数"""
    glPushMatrix()
    # 这里的y+height/2是为了让y坐标代表立方体的底部
    glTranslatef(x, y + height/2, z)
    glColor3f(*color)
    
    half_w, half_h, half_d = width/2, height/2, depth/2

    glBegin(GL_QUADS)
    # Front
    glVertex3f(-half_w, -half_h,  half_d); glVertex3f( half_w, -half_h,  half_d)
    glVertex3f( half_w,  half_h,  half_d); glVertex3f(-half_w,  half_h,  half_d)
    # Back
    glVertex3f(-half_w, -half_h, -half_d); glVertex3f(-half_w,  half_h, -half_d)
    glVertex3f( half_w,  half_h, -half_d); glVertex3f( half_w, -half_h, -half_d)
    # Top
    glVertex3f(-half_w,  half_h, -half_d); glVertex3f(-half_w,  half_h,  half_d)
    glVertex3f( half_w,  half_h,  half_d); glVertex3f( half_w,  half_h, -half_d)
    # Bottom
    glVertex3f(-half_w, -half_h, -half_d); glVertex3f( half_w, -half_h, -half_d)
    glVertex3f( half_w, -half_h,  half_d); glVertex3f(-half_w, -half_h,  half_d)
    # Right
    glVertex3f( half_w, -half_h, -half_d); glVertex3f( half_w,  half_h, -half_d)
    glVertex3f( half_w,  half_h,  half_d); glVertex3f( half_w, -half_h,  half_d)
    # Left
    glVertex3f(-half_w, -half_h, -half_d); glVertex3f(-half_w, -half_h,  half_d)
    glVertex3f(-half_w,  half_h,  half_d); glVertex3f(-half_w,  half_h, -half_d)
    glEnd()
    glPopMatrix()

def draw_text_2d(text, x, y, font_size, color):
    """在2D正交投影下绘制文本"""
    font = pygame.font.Font(None, font_size)
    text_surface = font.render(text, True, (int(color[0]*255), int(color[1]*255), int(color[2]*255)))
    text_data = pygame.image.tostring(text_surface, "RGBA", True)
    
    glPushAttrib(GL_ALL_ATTRIB_BITS)
    glDisable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    gluOrtho2D(0, DISPLAY_WIDTH, DISPLAY_HEIGHT, 0)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glRasterPos2d(x, y)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glDrawPixels(text_surface.get_width(), text_surface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glPopAttrib()