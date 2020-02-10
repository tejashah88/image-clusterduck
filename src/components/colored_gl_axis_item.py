# Source: http://www.pyqtgraph.org/documentation/_modules/pyqtgraph/opengl/items/GLAxisItem.html

import numpy as np

from OpenGL.GL import *
import pyqtgraph.opengl as gl

class ColoredGLAxisItem(gl.GLAxisItem):
    ''' A sub-class of GLAxisItem with the ability to customize the axes colors. '''

    def __init__(self, x_color=(255, 0, 0), y_color=(0, 255, 0), z_color=(0, 0, 255), **kwargs):
        super().__init__(**kwargs)
        self.x_color = x_color
        self.y_color = y_color
        self.z_color = z_color

    def paint(self):
        self.setupGLState()

        if self.antialias:
            glEnable(GL_LINE_SMOOTH)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        glBegin( GL_LINES )

        x, y, z = self.size()

        _z_color = (np.array(self.z_color) / 255).tolist()
        glColor4f(*_z_color, .6)  # z is blue by default
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, z)

        _y_color = (np.array(self.y_color) / 255).tolist()
        glColor4f(*_y_color, .6)  # y is green by default
        glVertex3f(0, 0, 0)
        glVertex3f(0, y, 0)

        _x_color = (np.array(self.x_color) / 255).tolist()
        glColor4f(*_x_color, .6)  # x is red by default
        glVertex3f(0, 0, 0)
        glVertex3f(x, 0, 0)
        glEnd()