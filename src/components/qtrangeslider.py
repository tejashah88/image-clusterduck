# Source: https://github.com/ThisIsClark/Qt-RangeSlider

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg


SC_HANDLE_SIDE_LENGTH = 11
SC_SLIDER_BAR_HEIGHT = 3
SC_LEFT_RIGHT_MARGIN = 1


BG_COLOR_ENABLED     = QtGui.QColor('#3daee9')
BG_COLOR_DISABLED    = QtGui.QColor('#4b535a')
OUTLINE_RECT_COLOR   = QtGui.QColor('#393d41')
HANDLE_COLOR         = QtGui.QColor('#464b50')
HANDLE_OUTLINE_COLOR = QtGui.QColor('#23272a')


class HandleOption:
    NoHandle = 0x0
    LeftHandle = 0x1
    RightHandle = 0x2
    DoubleHandles = LeftHandle | RightHandle

class QRangeSlider(QtGui.QWidget):
    rangeChanged = QtCore.pyqtSignal(int, int)
    valueChanged = QtCore.pyqtSignal(int, int)
    lowerValueChanged = QtCore.pyqtSignal(int)
    upperValueChanged = QtCore.pyqtSignal(int)

    def __init__(self, orientation=QtCore.Qt.Horizontal, range_type=HandleOption.DoubleHandles, parent=None):
        super().__init__(parent=parent)

        self._minimum = 0
        self._maximum = 100
        self._lower_val = 0
        self._upper_val = 100
        self.first_handle_pressed = False
        self.second_handle_pressed = False
        self.delta = 0
        self.bg_color = BG_COLOR_ENABLED
        self.orientation = orientation
        self.range_type = range_type

        self.setMouseTracking(True)

    def paintEvent(self, pevent):
        painter = QtGui.QPainter(self)

        # Background
        if self.orientation == QtCore.Qt.Horizontal:
            bg_rect = QtCore.QRectF(SC_LEFT_RIGHT_MARGIN, (self.height() - SC_SLIDER_BAR_HEIGHT) / 2, self.width() - SC_LEFT_RIGHT_MARGIN * 2, SC_SLIDER_BAR_HEIGHT)
        else:
            bg_rect = QtCore.QRectF((self.width() - SC_SLIDER_BAR_HEIGHT) / 2, SC_LEFT_RIGHT_MARGIN, SC_SLIDER_BAR_HEIGHT, self.height() - SC_LEFT_RIGHT_MARGIN*2)

        painter.setPen(QtGui.QPen(QtCore.Qt.black, 0.8))
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setBrush(QtGui.QBrush(OUTLINE_RECT_COLOR))
        painter.drawRoundedRect(bg_rect, 1, 1)
        painter.fillRect(bg_rect, QtGui.QBrush(OUTLINE_RECT_COLOR))

        # First value handle rect
        pen = QtGui.QPen(HANDLE_OUTLINE_COLOR, 0.8)
        painter.setPen(pen)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setBrush(QtGui.QBrush(HANDLE_COLOR))

        left_handle_rect = self.first_handle_rect
        if self.range_type & HandleOption.LeftHandle:
            painter.drawRoundedRect(left_handle_rect, 2, 2)

        # Second value handle rect
        right_handle_rect = self.second_handle_rect
        if self.range_type & HandleOption.RightHandle:
            painter.drawRoundedRect(right_handle_rect, 2, 2)

        # Handles
        selected_rect = QtCore.QRectF(bg_rect)
        if self.orientation == QtCore.Qt.Horizontal:
            selected_rect.setLeft((left_handle_rect.right() if self.range_type & HandleOption.LeftHandle else left_handle_rect.left()) + 0.5)
            selected_rect.setRight((right_handle_rect.left() if self.range_type & HandleOption.RightHandle else right_handle_rect.right()) - 0.5)
        else:
            selected_rect.setTop((left_handle_rect.bottom() if self.range_type & HandleOption.LeftHandle else left_handle_rect.top()) + 0.5)
            selected_rect.setBottom((right_handle_rect.top() if self.range_type & HandleOption.RightHandle else right_handle_rect.bottom()) - 0.5)

        if self.isEnabled():
            painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QBrush(self.bg_color))
        painter.drawRect(selected_rect)


    @property
    def first_handle_rect(self):
        percentage = (self._lower_val - self._minimum) * 1.0 / self.interval
        return self.handle_rect(percentage * self.valid_length() + SC_LEFT_RIGHT_MARGIN)


    @property
    def second_handle_rect(self):
        percentage = (self._upper_val - self._minimum) * 1.0 / self.interval
        return self.handle_rect(percentage * self.valid_length() + SC_LEFT_RIGHT_MARGIN + (SC_HANDLE_SIDE_LENGTH if self.range_type & HandleOption.DoubleHandles else 0))


    def handle_rect(self, val):
        if self.orientation == QtCore.Qt.Horizontal:
            return QtCore.QRectF(val, (self.height() - SC_HANDLE_SIDE_LENGTH) / 2, SC_HANDLE_SIDE_LENGTH, SC_HANDLE_SIDE_LENGTH)
        else:
            return QtCore.QRectF((self.width() - SC_HANDLE_SIDE_LENGTH) / 2, val, SC_HANDLE_SIDE_LENGTH, SC_HANDLE_SIDE_LENGTH)


    def mousePressEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton:
            pos_check = event.pos().y() if self.orientation == QtCore.Qt.Horizontal else event.pos().x()
            pos_max   = self.height()   if self.orientation == QtCore.Qt.Horizontal else self.width()
            pos_value = event.pos().x() if self.orientation == QtCore.Qt.Horizontal else event.pos().y()
            first_handle_rect_pos_value  = self.first_handle_rect.x()  if (self.orientation == QtCore.Qt.Horizontal) else self.first_handle_rect.y()
            second_handle_rect_pos_value = self.second_handle_rect.x() if (self.orientation == QtCore.Qt.Horizontal) else self.second_handle_rect.y()

            self.second_handle_pressed = self.second_handle_rect.contains(event.pos())
            self.first_handle_pressed  = not self.second_handle_pressed and self.first_handle_rect.contains(event.pos())

            if self.first_handle_pressed:
                self.delta = pos_value - (first_handle_rect_pos_value + SC_HANDLE_SIDE_LENGTH / 2)
            elif self.second_handle_pressed:
                self.delta = pos_value - (second_handle_rect_pos_value + SC_HANDLE_SIDE_LENGTH / 2)

            if pos_check >= 2 and pos_check <= pos_max - 2:
                step = 1 if self.interval / 10 < 1 else self.interval / 10
                if pos_value < first_handle_rect_pos_value:
                    self.lower_val = self.lower_val - step
                elif ((pos_value > first_handle_rect_pos_value + SC_HANDLE_SIDE_LENGTH) or not (self.range_type & HandleOption.LeftHandle)) and ((pos_value < second_handle_rect_pos_value) or not (self.range_type & HandleOption.RightHandle)):
                    if self.range_type == HandleOption.DoubleHandles:
                        if pos_value - (first_handle_rect_pos_value + SC_HANDLE_SIDE_LENGTH) < (second_handle_rect_pos_value - (first_handle_rect_pos_value + SC_HANDLE_SIDE_LENGTH)) / 2:
                            self.lower_val = self.lower_val + step if (self.lower_val + step < self.upper_val) else self.upper_val
                        else:
                            self.upper_val = self.upper_val - step if (self.upper_val - step > self.lower_val) else self.lower_val
                    elif self.range_type & HandleOption.LeftHandle:
                        self.lower_val = self.lower_val + step if (self.lower_val + step < self.upper_val) else self.upper_val
                    elif self.range_type & HandleOption.RightHandle:
                        self.upper_val = self.upper_val - step if (self.upper_val - step > self.lower_val) else self.lower_val
                elif pos_value > second_handle_rect_pos_value + SC_HANDLE_SIDE_LENGTH:
                    self.upper_val = self.upper_val + step


    def mouseMoveEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton:
            pos_value                    = event.pos().x() if self.orientation == QtCore.Qt.Horizontal else event.pos().y()
            first_handle_rect_pos_value  = self.first_handle_rect.x() if (self.orientation == QtCore.Qt.Horizontal) else self.first_handle_rect.y()
            second_handle_rect_pos_value = self.second_handle_rect.x() if (self.orientation == QtCore.Qt.Horizontal) else self.second_handle_rect.y()

            if self.first_handle_pressed and self.range_type & HandleOption.LeftHandle:
                if (pos_value - self.delta + SC_HANDLE_SIDE_LENGTH / 2) <= second_handle_rect_pos_value:
                    self.lower_val = (pos_value - self.delta - SC_LEFT_RIGHT_MARGIN - SC_HANDLE_SIDE_LENGTH / 2) * 1.0 / self.valid_length() * self.interval + self.minimum
                else:
                    self.lower_val = self.upper_val
            elif self.second_handle_pressed and self.range_type & HandleOption.RightHandle:
                if first_handle_rect_pos_value + SC_HANDLE_SIDE_LENGTH * (1.5 if self.range_type == HandleOption.DoubleHandles else 0.5) <= pos_value - self.delta:
                    self.upper_val = (pos_value - self.delta - SC_LEFT_RIGHT_MARGIN - SC_HANDLE_SIDE_LENGTH / 2 - (SC_HANDLE_SIDE_LENGTH if self.range_type == HandleOption.DoubleHandles else 0)) * 1.0 / self.valid_length() * self.interval + self.minimum
                else:
                    self.upper_val = self.lower_val


    def mouseReleaseEvent(self, event):
        self.first_handle_pressed = False
        self.second_handle_pressed = False


    def changeEvent(self, event):
        if event.type() == QtCore.QEvent.EnabledChange:
            if self.isEnabled():
                self.bg_color = BG_COLOR_ENABLED
            else:
                self.bg_color = BG_COLOR_DISABLED
            self.update()


    def minimumSizeHint(self):
        return QtCore.QSize(SC_HANDLE_SIDE_LENGTH * 2 + SC_LEFT_RIGHT_MARGIN * 2, SC_HANDLE_SIDE_LENGTH)


    @property
    def minimum(self):
        return self._minimum


    @minimum.setter
    def minimum(self, val):
        if val <= self._maximum:
            self._minimum = val
        else:
            old_max = self._maximum
            self._minimum = old_max
            self._maximum = val

        self.update()

        self.lower_val = self._minimum
        self.upper_val = self._maximum

        self.rangeChanged.emit(self.minimum, self.maximum)


    @property
    def maximum(self):
        return self._maximum


    @maximum.setter
    def maximum(self, val):
        if val >= self._minimum:
            self._maximum = val
        else:
            old_min = self._minimum
            self._maximum = old_min
            self._minimum = val

        self.update()

        self.lower_val = self._minimum
        self.upper_val = self._maximum

        self.rangeChanged.emit(self.minimum, self.maximum)


    @property
    def range(self):
        return (self.minimum, self.maximum)


    @range.setter
    def range(self, val):
        (self.minimum, self.maximum) = val


    @property
    def interval(self):
        return self.maximum - self.minimum


    @property
    def lower_val(self):
        return self._lower_val


    @lower_val.setter
    def lower_val(self, val):
        if val > self._maximum:
            val = self._maximum

        if val < self._minimum:
            val = self._minimum


        self._lower_val = val
        self.valueChanged.emit(self.lower_val, self.upper_val)
        self.lowerValueChanged.emit(self.lower_val)

        self.update()


    @property
    def upper_val(self):
        return self._upper_val


    @upper_val.setter
    def upper_val(self, val):
        if val > self._maximum:
            val = self._maximum

        if val < self._minimum:
            val = self._minimum

        self._upper_val = val
        self.valueChanged.emit(self.lower_val, self.upper_val)
        self.upperValueChanged.emit(self.upper_val)

        self.update()


    @property
    def values(self):
        return (self.lower_val, self.upper_val)


    @values.setter
    def values(self, val):
        (self.lower_val, self.upper_val) = val


    def valid_length(self):
        length = self.width() if self.orientation == QtCore.Qt.Horizontal else self.height()
        return length - SC_LEFT_RIGHT_MARGIN * 2 - SC_HANDLE_SIDE_LENGTH * (2 if self.range_type & HandleOption.DoubleHandles else 1)

if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        app = pg.mkQApp()

        MainWindow = QtGui.QMainWindow()

        widget = QtGui.QWidget(MainWindow)
        rsH = QRangeSlider(QtCore.Qt.Horizontal, HandleOption.DoubleHandles)
        rsV = QRangeSlider(QtCore.Qt.Vertical, HandleOption.DoubleHandles)
        rsHsingleLeft = QRangeSlider(QtCore.Qt.Horizontal, HandleOption.LeftHandle)
        rsVsingleLeft = QRangeSlider(QtCore.Qt.Vertical, HandleOption.LeftHandle)
        rsHsingleRight = QRangeSlider(QtCore.Qt.Horizontal, HandleOption.RightHandle)
        rsVsingleRight = QRangeSlider(QtCore.Qt.Vertical, HandleOption.RightHandle)

        palette = app.palette()

        def get_color(palette, state, role):
            state_enum = getattr(QtGui.QPalette, state)
            role_enum = getattr(QtGui.QPalette, role)
            return QtGui.QColor(palette.color(state_enum, role_enum))

        ALL_STATES  = [ 'Disabled', 'Active', 'Inactive', 'Normal' ]
        ALL_ROLES   = [ 'Window', 'Background', 'WindowText', 'Foreground',
                        'Base', 'AlternateBase', 'ToolTipBase', 'ToolTipText',
                        'Text', 'Button', 'ButtonText', 'BrightText', 'Highlight',
                        'HighlightedText', 'Link', 'LinkVisited' ]

        for state in ALL_STATES:
            for role in ALL_ROLES:
                color = get_color(palette, 'Active', role).name()
                css_str = state + ':' + role + ' { color: ' + color + ' }'
                print(css_str)


        layout = QtGui.QHBoxLayout()
        layout.addWidget(rsH)
        layout.addWidget(rsV)
        layout.addWidget(rsHsingleLeft)
        layout.addWidget(rsVsingleLeft)
        layout.addWidget(rsHsingleRight)
        layout.addWidget(rsVsingleRight)

        widget.setLayout(layout)
        widget.setStyleSheet('background-color:black')
        MainWindow.resize(QtGui.QDesktopWidget().availableGeometry(MainWindow).size() * 0.7)

        MainWindow.setCentralWidget(widget)
        MainWindow.show()

        sys.exit(app.exec_())
