from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg


class GuiBusyLock(object):
    ''' A object class to control GUI app interactions during long-term operations. '''

    def __init__(self, parent):
        self.parent = parent
        self.busy_cursor = pg.BusyCursor()

    def __enter__(self):
        self.parent.setWindowModality(QtCore.Qt.ApplicationModal)

        self.busy_cursor.__enter__()

    def __exit__(self, *args):
        self.busy_cursor.__exit__(*args)

        self.parent.setWindowModality(QtCore.Qt.NonModal)
        QtGui.QGuiApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
