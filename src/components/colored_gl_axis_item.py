import numpy as np

import pyqtgraph.opengl as gl

class ColoredGLAxisItem(gl.GLAxisItem):
    ''' A sub-class of GLAxisItem with the ability to customize the axes colors. '''

    def __init__(self, x_color=(255, 0, 0), y_color=(0, 255, 0), z_color=(0, 0, 255), **kwargs):
        # Set colors before super().__init__() since it calls updateLines()
        self.x_color = x_color
        self.y_color = y_color
        self.z_color = z_color
        super().__init__(**kwargs)

    def updateLines(self):
        if self.lineplot is None:
            return

        x, y, z = self.size()

        _z = (np.array(self.z_color) / 255).tolist()
        _y = (np.array(self.y_color) / 255).tolist()
        _x = (np.array(self.x_color) / 255).tolist()

        pos = np.array([
            [0, 0, 0,  0, 0, z],   # z axis
            [0, 0, 0,  0, y, 0],   # y axis
            [0, 0, 0,  x, 0, 0],   # x axis
        ], dtype=np.float32).reshape((-1, 3))

        color = np.array([
            [*_z, 0.6],
            [*_y, 0.6],
            [*_x, 0.6],
        ], dtype=np.float32)
        # Repeat color for both vertices of each line segment
        color = np.hstack((color, color)).reshape((-1, 4))

        self.lineplot.setData(pos=pos, color=color)
        self.update()
