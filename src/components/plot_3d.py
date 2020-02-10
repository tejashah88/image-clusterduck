from pyqtgraph.Qt import QtGui
import pyqtgraph.opengl as gl
from .colored_gl_axis_item import ColoredGLAxisItem

DEFAULT_AXIS_LENGTH = 3

class Plot3D(gl.GLViewWidget):
    def __init__(self, size=(600, 450), plot=None, enable_axes=True, axis_length=DEFAULT_AXIS_LENGTH):
        # Create and initialize 3D plotting widget
        super().__init__()

        # Create vars for the axes and 3d plot items
        self.axes_item = None
        self.plt_item = None
        self.cplt_item = None

        # Setup the title and plot size
        self.setFixedSize(*size)

        if enable_axes:
            self.enable_axes(axis_length)

        if plot is not None:
            self.set_plot(plot)


    def set_plot(self, plot):
        if self.plt_item is not None:
            self.plt_item.setData(pos=plot.pos, color=plot.color)
        else:
            self.plt_item = plot
            self.addItem(self.plt_item)


    def set_cluster_plot(self, cplot):
        if self.cplt_item is not None:
            self.cplt_item.setData(pos=cplot.pos, color=cplot.color)
        else:
            self.cplt_item = cplot
            self.addItem(self.cplt_item)


    def remove_cluster_plot(self):
        if self.cplt_item is not None:
            self.removeItem(self.cplt_item)
            self.cplt_item = None


    def enable_axes(self, axis_length=3):
        if self.axes_item is None:
            # x = blue, y = yellow, z = green
            self.axes_item = ColoredGLAxisItem(size=QtGui.QVector3D(1, 1, 1) * axis_length)
            self.addItem(self.axes_item)


    def disable_axes(self):
        if self.axes_item is not None:
            self.removeItem(self.axes_item)
            self.axes_item = None
