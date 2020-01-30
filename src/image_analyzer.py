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
from gui_busy_lock import GuiBusyLock

DEFAULT_IMG_FILENAME = './test-images/starry-night.jpg'
SUPPORTED_IMG_EXTS = '*.png *.jpg *.jpeg *.gif *.bmp *.tiff *.tif'

def img_scatterplot(cv_img, color_mode, crop_bounds=None, scale_factor=3):
    rgb_img = cv_img.RGB
    converted_img = cv_img[color_mode]

    if crop_bounds is not None:
        x_min, y_min, x_max, y_max = crop_bounds
    else:
        height, width = rgb_img.shape[:2]
        x_min, y_min, x_max, y_max = (0, 0, width, height)

    rgb_img = rgb_img[y_min:y_max, x_min:x_max]
    converted_img = converted_img[y_min:y_max, x_min:x_max]

    pos_arr = converted_img.reshape(-1, 3) / 255 * scale_factor
    color_arr = rgb_img.reshape(-1, 3) / 255

    return gl.GLScatterPlotItem(
        pos=pos_arr, color=color_arr,
        size=1, pxMode=True,
        glOptions='opaque'
    )


def pos_color_scatterplot(cv_img, color_mode, ch_index, crop_bounds=None, scale_factor=3):
    rgb_img = cv_img.RGB
    converted_img = cv_img[color_mode]

    if crop_bounds is not None:
        x_min, y_min, x_max, y_max = crop_bounds
    else:
        height, width = rgb_img.shape[:2]
        x_min, y_min, x_max, y_max = (0, 0, width, height)

    rgb_img = rgb_img[y_min:y_max, x_min:x_max]
    converted_img = converted_img[y_min:y_max, x_min:x_max]

    rows, cols = converted_img.shape[:2]
    c_arr, r_arr = np.meshgrid(np.arange(cols), np.arange(rows))
    channel_arr = converted_img[:, :, ch_index]

    scaled_dim = scale_factor / max(rows, cols)
    scaled_z = (scale_factor / 2) / 255

    row_array = (r_arr.flatten() - rows // 2) * scaled_dim
    col_array = (c_arr.flatten() - cols // 2) * scaled_dim
    ch_array = channel_arr.flatten() * scaled_z
    pos_arr = np.vstack( (row_array, col_array, ch_array) ).T

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

        self.menubar = None
        self.statusbar = None

        self.loading = False

        self.apply_crop = False


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


    @property
    def curr_image(self):
        return self.cv_img[self.color_mode]


    @property
    def curr_image_slice(self):
        return self.cv_img[self.color_mode][:, :, self.ch_index]


    def load_image(self, img_path, max_pixels=1000000):
        with GuiBusyLock(self):
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
                self.glvw_color_vis.set_plot(img_scatterplot(self.cv_img, self.color_mode))
                self.on_color_space_change(self.cs_index)


    def setup_gui(self):
        if self.cv_img is None:
            raise Exception('Error: Image has not been loaded yet! Please call load_image() before calling setup_gui()')

        # Setup main plots
        self.orig_img_plot = ImagePlotter(title='Original Image', img=self.cv_img.RGB, enable_roi=True)
        self.glvw_color_vis = Plot3D(plot=img_scatterplot(self.cv_img, self.color_mode))

        self.channel_plot = ImagePlotter(title=self.channel_mode, img=self.curr_image_slice)
        self.glvw_channel_vis = Plot3D(plot=pos_color_scatterplot(self.cv_img, self.color_mode, self.ch_index))

        self.cropped_img_plot = ImagePlotter(title='Cropped Image', img=self.cv_img.RGB)

        self.orig_img_plot.roi_item.sigRegionChanged.connect(self.on_crop_img_by_roi)
        setup_axes_links(self.orig_img_plot, [self.channel_plot])

        # Setup color space combo box
        self.color_space_cbox = QtGui.QComboBox()
        self.color_space_cbox.addItems(ALL_COLOR_SPACES)
        self.color_space_cbox.currentIndexChanged.connect(self.on_color_space_change)

        # Setup channel combo box
        self.channel_cbox = QtGui.QComboBox()
        self.channel_cbox.addItems(COLOR_SPACE_LABELS[self.color_mode])
        self.channel_cbox.currentIndexChanged.connect(self.on_channel_view_change)

        # Setup 'apply crop' checkbox
        self.apply_crop_box = QtGui.QCheckBox('Apply Crop')
        self.apply_crop_box.setChecked(self.apply_crop)
        self.apply_crop_box.toggled.connect(self.on_apply_crop_toggle)

        # Setup widgets according to given grid layout
        grid_layout = QtGui.QGridLayout()

        grid_layout.addWidget(self.color_space_cbox, 0, 0)
        grid_layout.addWidget(self.channel_cbox, 0, 1)
        grid_layout.addWidget(self.apply_crop_box, 0, 2)

        grid_layout.addWidget(self.orig_img_plot, 1, 0)
        grid_layout.addWidget(self.glvw_color_vis, 2, 0)

        grid_layout.addWidget(self.channel_plot, 1, 1)
        grid_layout.addWidget(self.glvw_channel_vis, 2, 1)

        grid_layout.addWidget(self.cropped_img_plot, 1, 2)

        # Set the layout and resize the window accordingly
        self.setLayout(grid_layout)
        self.resize(grid_layout.sizeHint() + QtCore.QSize(10, 30))


    def bind_to_main_window(self, main_window):
        main_window.setCentralWidget(self)

        self.setup_menubar(main_window)
        self.setup_statusbar(main_window)

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
        with GuiBusyLock(self):
            self.cs_index = cspace_index

            self.glvw_color_vis.set_plot(plot=img_scatterplot(self.cv_img, self.color_mode))

            self.channel_plot.setTitle(title=self.channel_mode)
            self.channel_plot.set_image(img=self.curr_image_slice)

            self.glvw_channel_vis.set_plot(plot=pos_color_scatterplot(self.cv_img, self.color_mode, self.ch_index))

            self.channel_cbox.clear()
            self.channel_cbox.addItems(COLOR_SPACE_LABELS[self.color_mode])

            self.on_channel_view_change(self.ch_index)


    def on_channel_view_change(self, ch_index):
        with GuiBusyLock(self):
            self.ch_index = ch_index

            # Update the title
            self.channel_plot.setTitle(self.channel_mode)

            # Update the image
            self.channel_plot.set_image(self.curr_image_slice)

            # Update the scatterplot
            self.on_apply_crop_toggle(self.apply_crop_box.isChecked())


    def on_apply_crop_toggle(self, should_apply_crop):
        self.apply_crop = should_apply_crop
        self.orig_img_plot.roi_item.sigRegionChanged.emit(self.orig_img_plot.roi_item)

        if not self.apply_crop:
            self.cropped_img_plot.set_image(self.cv_img.RGB)
            self.glvw_color_vis.set_plot(plot=img_scatterplot(self.cv_img, self.color_mode))

    def on_crop_img_by_roi(self, roi):
        if self.apply_crop:
            height, width = self.cv_img.RGB.shape[:2]
            x, y, w, h = roi.parentBounds().toAlignedRect().getRect()
            x_min, y_min = max(x, 0), max(y, 0)
            x_max, y_max = min(x + w, width), min(y + h, height)
            bounds = (x_min, y_min, x_max, y_max)

            self.cropped_img_plot.set_image(self.cv_img.RGB[y_min:y_max, x_min:x_max])
            self.glvw_color_vis.set_plot(plot=img_scatterplot(self.cv_img, self.color_mode, crop_bounds=bounds))
            self.glvw_channel_vis.set_plot(pos_color_scatterplot(self.cv_img, self.color_mode, self.ch_index, crop_bounds=bounds))


    def setup_menubar(self, main_window):
        self.menubar = main_window.menuBar()
        file_menu = self.menubar.addMenu('File')
        edit_menu = self.menubar.addMenu('Edit')
        view_menu = self.menubar.addMenu('View')
        tools_menu = self.menubar.addMenu('Tools')
        help_menu = self.menubar.addMenu('Help')

        open_image_action = QtGui.QAction('Open Image', self)
        open_image_action.setShortcut('Ctrl+O')
        open_image_action.setStatusTip('Open Image')
        open_image_action.triggered.connect(lambda: self.load_image(self.open_image_file_dialog()))
        file_menu.addAction(open_image_action)

        exit_action = QtGui.QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(main_window.close)
        file_menu.addAction(exit_action)


    def setup_statusbar(self, main_window):
        self.statusbar = main_window.statusBar()


    def show_status(self, text, timeout=0):
        self.statusbar.showMessage(text, timeout)


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
