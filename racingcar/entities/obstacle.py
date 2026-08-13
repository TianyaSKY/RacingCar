# entities/obstacle.py
import random

from ..utils import draw_cube
from ..config import *
from OpenGL.GL import glColor3f, glPopMatrix, glPushMatrix, glRotatef, glTranslatef
from OpenGL.GLU import GLU_SMOOTH, gluCylinder, gluDisk, gluNewQuadric, gluQuadricNormals


class Obstacle:
    """Visual obstacles that retain the original AABB collision contract."""
    def __init__(self, track_instance):
        self.track = track_instance
        self._quadric = None
        self.respawn()

    def respawn(self):
        """Choose a visual style without changing placement or collision dimensions."""
        self.kind = random.choice(("barricade", "cone"))
        self.passed = False
        tx, tz, nx, nz = self.track.get_random_track_position()

        offset = random.uniform(-ROAD_WIDTH * 0.8, ROAD_WIDTH * 0.8)
        self.x = tx + nx * offset
        self.z = tz + nz * offset

        self.width = random.uniform(1.0, 1.5)
        self.depth = random.uniform(1.0, 1.5)
        self.height = random.uniform(1.5, 2.5)

    def draw(self):
        """Draw the selected style while keeping the collision AABB unchanged."""
        charcoal = (0.035, 0.040, 0.050)
        draw_cube(
            self.x, -0.1, self.z,
            self.width * 1.2, 0.05, self.depth * 1.2, charcoal)

        if self.kind == "barricade":
            self._draw_barricade()
        elif self.kind == "cone":
            self._draw_cone()
        else:
            raise ValueError(f"Unknown obstacle kind: {self.kind}")

    def _draw_barricade(self):
        """Draw a layered construction barricade inside the existing AABB."""
        charcoal = (0.035, 0.040, 0.050)
        orange = (1.0, 0.31, 0.035)
        reflector = (1.0, 0.82, 0.32)
        amber = (1.0, 0.55, 0.06)

        base_height = 0.12
        foot_width = self.width * 0.18
        foot_height = self.height * 0.10
        foot_depth = self.depth * 0.72
        post_width = self.width * 0.08
        post_height = self.height * 0.84
        post_depth = self.depth * 0.16
        board_width = self.width * 0.92
        board_height = self.height * 0.38
        board_depth = self.depth * 0.34
        board_center_y = self.height * 0.48
        board_y = board_center_y - board_height / 2
        bar_height = self.height * 0.045
        bar_depth = self.depth * 0.025

        draw_cube(
            self.x, 0, self.z, self.width, base_height, self.depth, charcoal)
        for side in (-1, 1):
            draw_cube(
                self.x + side * self.width * 0.32, base_height, self.z,
                foot_width, foot_height, foot_depth, orange)
            draw_cube(
                self.x + side * self.width * 0.34, base_height, self.z,
                post_width, post_height, post_depth, charcoal)

        draw_cube(
            self.x, board_y, self.z,
            board_width, board_height, board_depth, orange)
        for bar_center_y in (self.height * 0.52, self.height * 0.69):
            for bar_z in (
                    self.z - board_depth / 2 + bar_depth / 2,
                    self.z + board_depth / 2 - bar_depth / 2):
                draw_cube(
                    self.x, bar_center_y - bar_height / 2, bar_z,
                    board_width, bar_height, bar_depth, reflector, shininess=72.0)

        lamp_size = min(self.width, self.depth) * 0.14
        draw_cube(
            self.x, board_y + board_height, self.z,
            lamp_size, self.height * 0.08, lamp_size, amber)

    def _draw_cone(self):
        """Draw a low-poly traffic cone inside the existing AABB."""
        charcoal = (0.035, 0.040, 0.050)
        orange = (1.0, 0.31, 0.035)
        reflector = (1.0, 0.82, 0.32)
        draw_cube(self.x, 0, self.z, self.width, 0.12, self.depth, charcoal)

        if self._quadric is None:
            self._quadric = gluNewQuadric()
            gluQuadricNormals(self._quadric, GLU_SMOOTH)

        base_radius = min(self.width, self.depth) * 0.32
        top_radius = base_radius * 0.18
        cone_height = self.height * 0.78
        band_y = cone_height * 0.64
        band_height = cone_height * 0.10
        taper = (top_radius - base_radius) / cone_height
        band_base_radius = base_radius + taper * band_y
        band_top_radius = base_radius + taper * (band_y + band_height)
        band_offset = min(self.width, self.depth) * 0.008

        glPushMatrix()
        glTranslatef(self.x, 0.12, self.z)
        # GLU cylinders grow along local +Z, so rotate them onto world +Y.
        glRotatef(-90.0, 1.0, 0.0, 0.0)
        glColor3f(*orange)
        gluCylinder(
            self._quadric, base_radius, top_radius, cone_height, 12, 2)

        glPushMatrix()
        glTranslatef(0.0, 0.0, band_y)
        glColor3f(*reflector)
        gluCylinder(
            self._quadric,
            band_base_radius + band_offset,
            band_top_radius + band_offset,
            band_height,
            12,
            1,
        )
        glPopMatrix()

        glTranslatef(0.0, 0.0, cone_height)
        glColor3f(*charcoal)
        gluDisk(self._quadric, 0.0, top_radius * 0.72, 12, 1)
        glPopMatrix()

    def get_bounds(self):
        return (self.x - self.width/2, self.x + self.width/2,
                self.z - self.depth/2, self.z + self.depth/2)
