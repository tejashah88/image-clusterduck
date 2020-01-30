import os

import numpy as np
import cv2

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from constants import ALL_COLOR_SPACES, COLOR_SPACE_LABELS
from cv_img import CvImg
from image_plotter import ImagePlotter
from plot_3d import Plot3D

DEFAULT_IMG_FILENAME = './test-images/starry-night.jpg'
SUPPORTED_IMG_EXTS = '*.png *.jpg *.jpeg *.gif *.bmp *.tiff *.tif'

def make_img_scatterplot(cv_img, color_mode, scale_factor=3):
    pos_arr = cv_img[color_mode].reshape(-1, 3) / 255 * scale_factor
    color_arr = cv_img.RGB.reshape(-1, 3) / 255

    return gl.GLScatterPlotItem(
        pos=pos_arr, color=color_arr,
        size=1, pxMode=True,
        glOptions='opaque'
    )


def make_pos_to_color_scatterplot(cv_img, color_mode, ch_index, scale_factor=3):
    rgb_img = cv_img.RGB
    converted_img = cv_img[color_mode]

    rows, cols = converted_img.shape[:2]
    r_arr, c_arr = np.mgrid[0:rows, 0:cols]
    channel_arr = converted_img[:, :, ch_index]

    pos_arr = np.vstack((r_arr.flatten(), c_arr.flatten(), channel_arr.flatten())).T
    pos_arr = pos_arr / np.array([ max(rows, cols), max(rows, cols), max(pos_arr[:, 2]) ])
    pos_arr = pos_arr * np.array([ scale_factor, scale_factor, scale_factor / 2 ])

    color_arr = rgb_img.reshape(-1, 3) / 255

    return gl.GLScatterPlotItem(
        pos=pos_arr, color=color_arr,
        size=1, pxMode=True,
        glOptions='opaque'
    )

# Link the image plot axes together for consistent panning and zooming
def setup_axes_links(leader_plot, follower_plots):
    for plot in follower_plots:
        plot.setXLink(leader_plot)
        plot.setYLink(leader_plot)

# Interpret image data as row-major instead of col-major
pg.setConfigOptions(imageAxisOrder='row-major')


class MyWindow(pg.GraphicsLayoutWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Image Cluster Analysis')

        self.cv_img = None
        self.ch_index = 0
        self.cs_index = 0

        self.orig_img_plot = None
        self.glvw_color_vis = None
        self.channel_plot = None
        self.glvw_channel_vis = None


    @property
    def gui_ready(self):
        # HACK: This only checks the 4 main plots
        return None not in [self.orig_img_plot, self.glvw_color_vis, self.channel_plot, self.glvw_channel_vis]


    @property
    def color_mode(self):
        return ALL_COLOR_SPACES[self.cs_index]


    @property
    def channel_mode(self):
        return COLOR_SPACE_LABELS[self.color_mode][self.ch_index]


    def load_image(self, img_path, max_pixels=1000000):
        input_img = cv2.imread(img_path)

        if input_img is None:
            print(f'Error: Unable to load image from {img_path}')
            exit(-1)

        height, width = input_img.shape[:2]
        num_pixels = width * height
        if num_pixels > max_pixels:
            resize_factor = 1 / ( (num_pixels / max_pixels) ** 0.5 )
            print('Resize factor:', resize_factor)
            input_img = cv2.resize(input_img, None, fx=resize_factor, fy=resize_factor)

        self.cv_img = CvImg.from_ndarray(input_img)

        print('Original number of pixels:', num_pixels)

        if self.gui_ready:
            self.orig_img_plot.set_image(self.cv_img.RGB)
            self.glvw_color_vis.set_plot(make_img_scatterplot(self.cv_img, self.color_mode))
            self.on_color_space_change(self.cs_index)
            self.on_channel_view_change(self.ch_index)


    def setup_gui(self):
        if self.cv_img is None:
            raise Exception('Error: Image has not been loaded yet! Please call load_image() before calling setup_gui()')

        self.orig_img_plot = ImagePlotter(title='Original Image', img=self.cv_img.RGB, enable_roi=True)
        self.glvw_color_vis = Plot3D(plot=make_img_scatterplot(self.cv_img, self.color_mode))

        channel_isolated_img = self.cv_img[self.color_mode][:, :, self.ch_index]
        self.channel_plot = ImagePlotter(title=self.channel_mode, img=channel_isolated_img)
        self.glvw_channel_vis = Plot3D(plot=make_pos_to_color_scatterplot(self.cv_img, self.color_mode, self.ch_index))

        setup_axes_links(self.orig_img_plot, [self.channel_plot])

        # Setup color space combo box
        self.color_space_cbox = QtGui.QComboBox()
        self.color_space_cbox.addItems(ALL_COLOR_SPACES)
        self.color_space_cbox.currentIndexChanged.connect(self.on_color_space_change)

        # Setup channel combo box
        self.channel_cbox = QtGui.QComboBox()
        self.channel_cbox.addItems(COLOR_SPACE_LABELS[self.color_mode])
        self.channel_cbox.currentIndexChanged.connect(self.on_channel_view_change)

        # Setup widgets according to given grid layout
        grid_layout = QtGui.QGridLayout()

        grid_layout.addWidget(self.color_space_cbox, 0, 0)
        grid_layout.addWidget(self.channel_cbox, 0, 1)

        grid_layout.addWidget(self.orig_img_plot, 1, 0)
        grid_layout.addWidget(self.glvw_color_vis, 2, 0)

        grid_layout.addWidget(self.channel_plot, 1, 1)
        grid_layout.addWidget(self.glvw_channel_vis, 2, 1)

        # Set the layout and resize the window accordingly
        self.setLayout(grid_layout)
        self.resize(grid_layout.sizeHint() + QtCore.QSize(10, 10))


    def bind_to_main_window(self, main_window):
        main_window.setCentralWidget(self)

        self.setup_menubar(main_window)

        main_window.resize(self.size())


    def open_image_file_dialog(self):
        home = os.path.expanduser('~')
        home = os.path.curdir # FIXME
        filename, _ = pg.FileDialog().getOpenFileName(
            self, 'Open image file', home,
            f'Image Files ({SUPPORTED_IMG_EXTS});;All Files (*)'
        )
        return filename


    def on_color_space_change(self, cspace_index):
        self.cs_index = cspace_index

        self.glvw_color_vis.set_plot(plot=make_img_scatterplot(self.cv_img, self.color_mode))

        self.channel_plot.setTitle(title=self.channel_mode)
        self.channel_plot.set_image(img=self.cv_img[self.color_mode][:, :, self.ch_index])

        self.glvw_channel_vis.set_plot(plot=make_pos_to_color_scatterplot(self.cv_img, self.color_mode, self.ch_index))

        self.channel_cbox.clear()
        self.channel_cbox.addItems(COLOR_SPACE_LABELS[self.color_mode])

        self.on_channel_view_change(self.ch_index)

    def on_channel_view_change(self, ch_index):
        new_title = COLOR_SPACE_LABELS[self.color_mode][ch_index]
        new_img = self.cv_img[self.color_mode][:, :, ch_index]

        # Update the title
        self.channel_plot.setTitle(new_title)

        # Update the image
        self.channel_plot.set_image(new_img)

        # Update the scatterplot
        new_scatter = make_pos_to_color_scatterplot(self.cv_img, self.color_mode, ch_index)
        self.glvw_channel_vis.plt_item.setData(pos=new_scatter.pos, color=new_scatter.color)


    def setup_menubar(self, main_window):
        menubar = main_window.menuBar()
        fileMenu = menubar.addMenu('File')
        editMenu = menubar.addMenu('Edit')
        viewMenu = menubar.addMenu('View')
        searchMenu = menubar.addMenu('Search')
        toolsMenu = menubar.addMenu('Tools')
        helpMenu = menubar.addMenu('Help')

        openImageAction = QtGui.QAction('Open Image', self)
        openImageAction.setShortcut('Ctrl+O')
        openImageAction.setStatusTip('Open Image')
        openImageAction.triggered.connect(lambda: self.load_image(self.open_image_file_dialog()))
        fileMenu.addAction(openImageAction)

        exitAction = QtGui.QAction('Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(main_window.close)
        fileMenu.addAction(exitAction)


    def setup_statusbar(self, main_window):
        statusbar = main_window.statusBar()

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        app = pg.mkQApp()
        MainWindow = QtGui.QMainWindow()

        gui = MyWindow()
        gui.load_image(DEFAULT_IMG_FILENAME)
        gui.setup_gui()
        gui.bind_to_main_window(MainWindow)
        MainWindow.show()

        sys.exit(app.exec_())
