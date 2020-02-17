# Source: http://www.pyqtgraph.org/documentation/_modules/pyqtgraph/opengl/items/GLGridItem.html

import numpy as np

from OpenGL.GL import *
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtGui

class GL3DGridItem(gl.GLGridItem):
    def __init__(self, position=None, **kwargs):
        super().__init__(**kwargs)

        self.setSize(x=20, y=20, z=0)
        if position is None:
            position = QtGui.QVector3D(0, 0, 0)
        self.setPosition(position=position)

    def setPosition(self, x=None, y=None, z=None, position=None):
        """
        Set the position of the axes (in its local coordinate system; this does not affect the transform)
        Arguments can be x,y,z or size=QVector3D().
        """
        if position is not None:
            x = position.x()
            y = position.y()
            z = position.z()
        self.__position = [x,y,z]
        self.update()


    def position(self):
        return self.__position[:]

    def paint(self):
        self.setupGLState()

        if self.antialias:
            glEnable(GL_LINE_SMOOTH)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        glBegin( GL_LINES )

        ps_x, ps_y, ps_z = self.position()
        sz_x, sz_y, sz_z = self.size()
        sp_x, sp_y, sp_z = self.spacing()

        xvals = np.arange(ps_x, ps_x + sz_x + sp_x*0.001, sp_x)
        yvals = np.arange(ps_y, ps_y + sz_y + sp_y*0.001, sp_y)
        zvals = np.arange(ps_z, ps_z + sz_z + sp_z*0.001, sp_z)

        glColor4f(1, 1, 1, .3)
        for x in xvals:
            # draw x-axis lines along y-axis
            glVertex3f(x, yvals[0], 0)
            glVertex3f(x, yvals[-1], 0)

            # draw x-axis lines along z-axis
            glVertex3f(x, 0, zvals[0])
            glVertex3f(x, 0, zvals[-1])

        for y in yvals:
            # draw y-axis lines along x-axis
            glVertex3f(xvals[0], y, 0)
            glVertex3f(xvals[-1], y, 0)

            # draw y-axis lines along z-axis
            glVertex3f(0, y, zvals[0])
            glVertex3f(0, y, zvals[-1])

        for z in zvals:
            # draw z-axis lines along x-axis
            glVertex3f(xvals[0], 0, z)
            glVertex3f(xvals[-1], 0, z)

            # draw z-axis lines along y-axis
            glVertex3f(0, yvals[0], z)
            glVertex3f(0, yvals[-1], z)

        glEnd()
