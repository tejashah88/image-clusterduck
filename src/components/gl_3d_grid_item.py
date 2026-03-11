import numpy as np

import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtGui

class GL3DGridItem(gl.GLGridItem):
    def __init__(self, position=None, **kwargs):
        if position is None:
            self._position = [0.0, 0.0, 0.0]
        else:
            self._position = [position.x(), position.y(), position.z()]
        super().__init__(**kwargs)

    def setPosition(self, x=None, y=None, z=None, position=None):
        """
        Set the position of the grid origin.
        Arguments can be x,y,z or position=QVector3D().
        """
        if position is not None:
            x = position.x()
            y = position.y()
            z = position.z()
        self._position = [x, y, z]
        self.updateLines()

    def position(self):
        return self._position[:]

    def updateLines(self):
        if self.lineplot is None:
            return

        ps_x, ps_y, ps_z = getattr(self, '_position', [0.0, 0.0, 0.0])
        sz_x, sz_y, sz_z = self.size()
        sp_x, sp_y, sp_z = self.spacing()

        xvals = np.arange(ps_x, ps_x + sz_x + sp_x * 0.001, sp_x)
        yvals = np.arange(ps_y, ps_y + sz_y + sp_y * 0.001, sp_y)
        zvals = np.arange(ps_z, ps_z + sz_z + sp_z * 0.001, sp_z)

        lines = []

        for x in xvals:
            # XY plane (z = ps_z)
            lines.append([x, yvals[0], ps_z,  x, yvals[-1], ps_z])
            # XZ plane (y = ps_y)
            if len(zvals) > 1:
                lines.append([x, ps_y, zvals[0],  x, ps_y, zvals[-1]])

        for y in yvals:
            # XY plane (z = ps_z)
            lines.append([xvals[0], y, ps_z,  xvals[-1], y, ps_z])
            # YZ plane (x = ps_x)
            if len(zvals) > 1:
                lines.append([ps_x, y, zvals[0],  ps_x, y, zvals[-1]])

        for z in zvals:
            # XZ plane (y = ps_y)
            if len(xvals) > 1:
                lines.append([xvals[0], ps_y, z,  xvals[-1], ps_y, z])
            # YZ plane (x = ps_x)
            if len(yvals) > 1:
                lines.append([ps_x, yvals[0], z,  ps_x, yvals[-1], z])

        pos = np.array(lines, dtype=np.float32).reshape((-1, 3))
        self.lineplot.setData(pos=pos, color=self.color())
        self.update()
