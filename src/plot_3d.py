from pyqtgraph.Qt import QtGui
import pyqtgraph.opengl as gl

class Plot3D(gl.GLViewWidget):
    def __init__(self, size=(600, 450), plot=None, enable_axes=True, axis_length=3):
        # Create and initialize 3D plotting widget
        super().__init__()

        # Create vars for the axes and 3d plot items
        self.axes_item = None
        self.plt_item = None

        # Setup the title and plot size
        # self.setTitle(title)
        self.setFixedSize(*size)

        if enable_axes:
            self.enable_axes(axis_length)

        if plot is not None:
            self.set_plot(plot)


    def set_plot(self, plt):
        if self.plt_item is not None:
            self.removeItem(self.plt_item)

        self.plt_item = plt
        self.addItem(self.plt_item)


    def enable_axes(self, axis_length=3):
        if self.axes_item is None:
            # x = blue, y = yellow, z = green
            self.axes_item = gl.GLAxisItem(size=QtGui.QVector3D(1, 1, 1) * axis_length)
            self.addItem(self.axes_item)


    def disable_axes(self):
        if self.axes_item is not None:
            self.removeItem(self.axes_item)
            self.axes_item = None
