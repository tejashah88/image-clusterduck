import pyqtgraph as pg

class GlobalDataTreeWidget(pg.DataTreeWidget):
    ''' A modded DataTreeWidget that makes it slightly easier to manage many signal events updating the data tree. '''
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.global_data = {}


    def __getitem__(self, key):
        return self.global_data[key]


    def __setitem__(self, key, value):
        self.global_data[key] = value
        self.update_data()


    def set_data(self, data={}):
        self.global_data = {**self.global_data, **data}
        self.update_data()


    def update_data(self):
        super().setData(self.global_data, hideRoot=True)
        self.shrink_columns_to_contents()


    def swap_key(self, old_key, new_key):
        val = self[old_key]
        del self[old_key]
        self[new_key] = val


    def shrink_columns_to_contents(self):
        self.resizeColumnToContents(0)
        self.resizeColumnToContents(1)
        self.resizeColumnToContents(2)
