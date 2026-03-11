import numpy as np
# from fdict import fdict
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

class GlobalDataTreeWidget(pg.DataTreeWidget):
    ''' A modded DataTreeWidget that makes it slightly easier to manage many signal events updating the data tree. '''
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.global_data = {}
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)


    def __getitem__(self, key):
        return self.global_data[key]


    def __setitem__(self, key, value):
        if key not in self.global_data:
            raise Exception(f'Error: "{key}" not found in global data tree')

        self.global_data[key] = value
        self.update_data()


    def __delitem__(self, key):
        del self.global_data[key]


    # Since pyqtgraph 0.11.0, they convert numpy arrays to TableViews, which is considerable slower
    # than printing stringified arrays instead
    def _stringify_numpy_arrays(self, _dict):
        for key in _dict.keys():
            if isinstance(_dict[key], np.ndarray):
                _dict[key] = str(_dict[key])


    def set_data(self, data={}):
        self.global_data = dict(data)
        self.update_data()


    def _content_height(self):
        row_h = self.sizeHintForRow(0)
        if row_h <= 0:
            row_h = 20
        it = QtWidgets.QTreeWidgetItemIterator(self, QtWidgets.QTreeWidgetItemIterator.NotHidden)
        count = 0
        while it.value():
            count += 1
            it += 1
        return self.header().height() + count * row_h

    def sizeHint(self):
        return QtCore.QSize(super().sizeHint().width(), self._content_height())

    def update_data(self):
        self._stringify_numpy_arrays(self.global_data)
        super().setData(self.global_data, hideRoot=True)
        self.shrink_columns_to_contents()
        self.setMinimumHeight(self._content_height())
        self.updateGeometry()


    def swap_key(self, old_key, new_key):
        val = self[old_key]
        del self[old_key]
        self[new_key] = val


    def contextMenuEvent(self, event):
        item = self.itemAt(self.viewport().mapFromGlobal(event.globalPos()))
        if item is None:
            return
        menu = QtWidgets.QMenu(self)
        copy_action = menu.addAction('Copy')
        if menu.exec_(event.globalPos()) == copy_action:
            text = item.text(2) or item.text(0)
            QtWidgets.QApplication.clipboard().setText(text)

    def shrink_columns_to_contents(self):
        self.resizeColumnToContents(0)
        self.resizeColumnToContents(1)
        self.resizeColumnToContents(2)
