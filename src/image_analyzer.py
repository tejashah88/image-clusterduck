import os

import numpy as np
import cv2

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl

import sklearn.cluster

from constants import *
from cv_img import CvImg
from image_plotter import ImagePlotter
from plot_3d import Plot3D
from gui_busy_lock import GuiBusyLock
from image_clusterers import KMeansImageClusterer, MiniBatchKMeansImageClusterer

DEFAULT_IMG_FILENAME = './test-images/starry-night.jpg'
SUPPORTED_IMG_EXTS = '*.png *.jpg *.jpeg *.gif *.bmp *.tiff *.tif'


IMG_CLUSTERERS = [
    KMeansImageClusterer(),
    MiniBatchKMeansImageClusterer(),
]


def create_cluster_plot(cv_img, color_mode):
    color_centers, error = kmeans_image(cv_img, color_mode, 10)
    colored_centers = get_rgb_from(color_centers, color_mode) / 255
    # color_centers = (color_centers * 255).astype(int)
    return gl.GLScatterPlotItem(
        pos=color_centers / 255 * 3, color=colored_centers,
        size=0.5, pxMode=not True,
        glOptions='opaque'
    )


def img_scatterplot(cv_img, color_mode, ch_index, crop_bounds=None, thresh_bounds=None, scale_factor=3):
    rgb_img = cv_img.RGB
    converted_img = cv_img[color_mode]

    if crop_bounds is not None:
        x_min, y_min, x_max, y_max = crop_bounds
    else:
        height, width = rgb_img.shape[:2]
        x_min, y_min, x_max, y_max = (0, 0, width, height)

    rgb_img = rgb_img[y_min:y_max, x_min:x_max]
    converted_img = converted_img[y_min:y_max, x_min:x_max]

    if thresh_bounds is not None:
        lower_ch, upper_ch = thresh_bounds
    else:
        lower_ch, upper_ch = (0, 255)

    channel_arr = converted_img[:, :, ch_index]
    rgb_img[( (channel_arr < lower_ch) | (channel_arr > upper_ch) )] = 0
    channel_arr[( (channel_arr < lower_ch) | (channel_arr > upper_ch) )] = 0

    pos_arr = converted_img.reshape(-1, 3) / 255 * scale_factor
    color_arr = rgb_img.reshape(-1, 3) / 255

    return gl.GLScatterPlotItem(
        pos=pos_arr, color=color_arr,
        size=1, pxMode=True,
        glOptions='opaque'
    )


def pos_color_scatterplot(cv_img, color_mode, ch_index, crop_bounds=None, thresh_bounds=None, scale_factor=5):
    rgb_img = cv_img.RGB
    converted_img = cv_img[color_mode]

    if crop_bounds is not None:
        x_min, y_min, x_max, y_max = crop_bounds
    else:
        height, width = rgb_img.shape[:2]
        x_min, y_min, x_max, y_max = (0, 0, width, height)

    rgb_img = rgb_img[y_min:y_max, x_min:x_max]
    converted_img = converted_img[y_min:y_max, x_min:x_max]

    if thresh_bounds is not None:
        lower_ch, upper_ch = thresh_bounds
    else:
        lower_ch, upper_ch = (0, 255)

    rows, cols = converted_img.shape[:2]
    c_arr, r_arr = np.meshgrid(np.arange(cols), np.arange(rows))
    channel_arr = converted_img[:, :, ch_index]

    thresh_indicies = ( (channel_arr < lower_ch) | (channel_arr > upper_ch) )
    rgb_img[thresh_indicies] = 0
    r_arr[thresh_indicies] = 0
    c_arr[thresh_indicies] = 0
    channel_arr[thresh_indicies] = 0

    scaled_dim = scale_factor / max(rows, cols)
    scaled_z = 2 / 255

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
        self.cluster_index = 0

        self.orig_img_plot = None
        self.glvw_color_vis = None
        self.channel_plot = None
        self.glvw_channel_vis = None

        self.roi = None
        self.menubar = None
        self.statusbar = None

        self.apply_crop = False
        self.apply_thresh = False
        self.apply_cluster = False
        self.apply_hist = False

        self.lower_thresh = 0
        self.upper_thresh = 255


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
        img_slice = self.cv_img[self.color_mode][:, :, self.ch_index]
        if self.apply_thresh:
            lower_ch, upper_ch = self.thresh_bounds
            thresh_indicies = ( (img_slice < lower_ch) | (img_slice > upper_ch) )
            img_slice[thresh_indicies] = 0
        return img_slice


    @property
    def roi_bounds(self):
        if self.roi is not None:
            height, width = self.cv_img.RGB.shape[:2]
            if self.apply_crop:
                x, y, w, h = self.roi.parentBounds().toAlignedRect().getRect()
                x_min, y_min = max(x, 0), max(y, 0)
                x_max, y_max = min(x + w, width), min(y + h, height)
                return (x_min, y_min, x_max, y_max)
            else:
                return (0, 0, width, height)
        return None


    @property
    def thresh_bounds(self):
        if self.apply_thresh:
            return (self.lower_thresh, self.upper_thresh)
        return None


    @thresh_bounds.setter
    def thresh_bounds(self, val):
        self.lower_thresh, self.upper_thresh = val


    @property
    def curr_img_scatterplot(self):
        return img_scatterplot(self.cv_img, self.color_mode, self.ch_index, crop_bounds=self.roi_bounds, thresh_bounds=self.thresh_bounds)


    @property
    def curr_pos_color_scatterplot(self):
        return pos_color_scatterplot(self.cv_img, self.color_mode, self.ch_index, crop_bounds=self.roi_bounds, thresh_bounds=self.thresh_bounds)


    def load_image(self, img_path, max_pixels=1000000):
        with GuiBusyLock(self):
            input_img = cv2.imread(img_path)

            if input_img is None:
                print(f'Error: Unable to load image from {img_path}')
                return

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
                self.on_color_space_change(self.cs_index)
                self.on_img_modify()


    def setup_gui(self):
        if self.cv_img is None:
            raise Exception('Error: Image has not been loaded yet! Please call load_image() before calling setup_gui()')

        # Setup main plots
        self.orig_img_plot = ImagePlotter(title='Original Image', img=self.cv_img.RGB, enable_roi=True)
        self.roi = self.orig_img_plot.roi_item
        self.glvw_color_vis = Plot3D(plot=self.curr_img_scatterplot)
        # self.glvw_color_vis.set_cluster_plot(create_cluster_plot(self.cv_img, self.color_mode))

        self.channel_plot = ImagePlotter(title=self.channel_mode, img=self.curr_image_slice)
        self.glvw_channel_vis = Plot3D(plot=self.curr_pos_color_scatterplot, enable_axes=False)

        self.cropped_img_plot = ImagePlotter(title='Cropped Image', img=self.cv_img.RGB)

        self.roi.sigRegionChanged.connect(self.on_crop_modify)
        setup_axes_links(self.orig_img_plot, [self.channel_plot])

        # Setup color space combo box
        self.color_space_cbox = QtGui.QComboBox()
        self.color_space_cbox.addItems(ALL_COLOR_SPACES)
        self.color_space_cbox.setCurrentIndex(self.cs_index)
        self.color_space_cbox.currentIndexChanged.connect(self.on_color_space_change)

        # Setup channel combo box
        self.channel_cbox = QtGui.QComboBox()
        self.channel_cbox.addItems(COLOR_SPACE_LABELS[self.color_mode])
        self.channel_cbox.setCurrentIndex(self.ch_index)
        self.channel_cbox.currentIndexChanged.connect(self.on_channel_view_change)

        # Setup crop, thresholding, clustering, and histogram checkboxes
        self.apply_crop_box = QtGui.QCheckBox('Apply Cropping')
        self.apply_crop_box.setChecked(self.apply_crop)
        self.apply_crop_box.toggled.connect(self.on_apply_crop_toggle)

        self.apply_thresh_box = QtGui.QCheckBox('Apply Thresholding')
        self.apply_thresh_box.setChecked(self.apply_thresh)
        self.apply_thresh_box.toggled.connect(self.on_apply_thresh_toggle)

        self.apply_cluster_box = QtGui.QCheckBox('Apply Clustering')
        self.apply_cluster_box.setChecked(self.apply_cluster)
        self.apply_cluster_box.toggled.connect(self.on_apply_cluster_toggle)

        self.apply_hist_box = QtGui.QCheckBox('Show Histograms')
        self.apply_hist_box.setChecked(self.apply_hist)
        self.apply_hist_box.toggled.connect(self.on_apply_hist_toggle)

        # Setup clustering algorithm combo box
        self.cluster_cbox = QtGui.QComboBox()
        self.cluster_cbox.addItems(ALL_CLUSTER_ALGORITHMS)
        self.cluster_cbox.setCurrentIndex(self.cluster_index)
        self.cluster_cbox.currentIndexChanged.connect(self.on_cluster_algo_change)
        self.cluster_cbox.setEnabled(False)

        # Setup thresholding sliders based on current channel
        self.channel_lower_thresh_slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.channel_lower_thresh_slider.setMinimum(0)
        self.channel_lower_thresh_slider.setMaximum(255)
        self.channel_lower_thresh_slider.setValue(0)
        self.channel_lower_thresh_slider.setEnabled(False)
        self.channel_lower_thresh_slider.valueChanged.connect(lambda lower_val: self.on_thresh_change('lower', lower_val))

        self.channel_upper_thresh_slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.channel_upper_thresh_slider.setMinimum(0)
        self.channel_upper_thresh_slider.setMaximum(255)
        self.channel_upper_thresh_slider.setValue(255)
        self.channel_upper_thresh_slider.setEnabled(False)
        self.channel_upper_thresh_slider.valueChanged.connect(lambda upper_val: self.on_thresh_change('upper', upper_val))

        # Setup cluster calculating button
        self.run_clustering_button = QtGui.QPushButton('Run Clustering')
        self.run_clustering_button.setEnabled(False)
        self.run_clustering_button.clicked.connect(self.on_run_clustering)

        # Setup widgets according to given grid layout
        grid_layout = QtGui.QGridLayout()

        grid_layout.addWidget(self.orig_img_plot, 0, 0)
        grid_layout.addWidget(self.glvw_color_vis, 1, 0)

        grid_layout.addWidget(self.channel_plot, 0, 1)
        grid_layout.addWidget(self.glvw_channel_vis, 1, 1)

        grid_layout.addWidget(self.cropped_img_plot, 0, 2)

        self.settings_grid_layout = QtGui.QGridLayout()

        self.settings_grid_layout.addWidget(QtGui.QLabel('Color Space:'), 0, 0)
        self.settings_grid_layout.addWidget(self.color_space_cbox, 0, 1)
        self.settings_grid_layout.addWidget(QtGui.QLabel('Channel:'), 1, 0)
        self.settings_grid_layout.addWidget(self.channel_cbox, 1, 1)

        self.settings_grid_layout.addWidget(self.apply_crop_box, 2, 0)
        self.settings_grid_layout.addWidget(self.apply_thresh_box, 2, 1)

        self.settings_grid_layout.addWidget(QtGui.QLabel('Lower Threshold:'), 3, 0)
        self.settings_grid_layout.addWidget(self.channel_lower_thresh_slider, 3, 1)
        self.settings_grid_layout.addWidget(QtGui.QLabel('Upper Threshold:'), 4, 0)
        self.settings_grid_layout.addWidget(self.channel_upper_thresh_slider, 4, 1)

        self.settings_grid_layout.addWidget(self.apply_cluster_box, 5, 0)
        self.settings_grid_layout.addWidget(self.apply_hist_box, 5, 1)

        self.settings_grid_layout.addWidget(QtGui.QLabel('Cluster Algorithm:'), 6, 0)
        self.settings_grid_layout.addWidget(self.cluster_cbox, 6, 1)

        self.clusterer_controller = IMG_CLUSTERERS[self.cluster_index]
        cluster_settings_layout = self.clusterer_controller.setup_settings_layout()

        self.cluster_settings_widget = QtGui.QWidget()
        self.cluster_settings_widget.setLayout(cluster_settings_layout)
        self.cluster_settings_widget.setEnabled(False)
        self.settings_grid_layout.addWidget(self.cluster_settings_widget, 7, 0, 1, 2)

        self.settings_grid_layout.addWidget(self.run_clustering_button, 8, 1)

        self.settings_grid_layout.addWidget(QtGui.QLabel(''), 99, 0)

        options_widget = QtGui.QWidget()
        options_widget.setLayout(self.settings_grid_layout)

        grid_layout.addWidget(options_widget, 1, 2)

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

            self.glvw_color_vis.set_plot(plot=self.curr_img_scatterplot)
            self.glvw_channel_vis.set_plot(plot=self.curr_pos_color_scatterplot)

            self.channel_cbox.clear()
            self.channel_cbox.addItems(COLOR_SPACE_LABELS[self.color_mode])

            self.on_channel_view_change(self.ch_index)


    def on_channel_view_change(self, ch_index):
        with GuiBusyLock(self):
            self.ch_index = ch_index

            # Update the title
            self.channel_plot.setTitle(title=self.channel_mode)

            # Update the image
            self.channel_plot.set_image(img=self.curr_image_slice)

            # Update the scatterplot
            self.glvw_channel_vis.set_plot(plot=self.curr_pos_color_scatterplot)


    def on_cluster_algo_change(self, cluster_index):
        self.cluster_index = cluster_index

        self.clusterer_controller = IMG_CLUSTERERS[self.cluster_index]
        cluster_settings_layout = self.clusterer_controller.setup_settings_layout()

        old_widget = self.cluster_settings_widget
        self.cluster_settings_widget = QtGui.QWidget()
        self.cluster_settings_widget.setLayout(cluster_settings_layout)
        self.cluster_settings_widget.setEnabled(self.apply_cluster)

        self.settings_grid_layout.replaceWidget(old_widget, self.cluster_settings_widget)
        QtCore.QObjectCleanupHandler().add(old_widget)
        self.settings_grid_layout.update()


    def on_crop_modify(self):
        if self.apply_crop:
            (x_min, y_min, x_max, y_max) = self.roi_bounds
            self.cropped_img_plot.set_image(self.cv_img.RGB[y_min:y_max, x_min:x_max])
            self.glvw_color_vis.set_plot(plot=self.curr_img_scatterplot)
            self.glvw_channel_vis.set_plot(plot=self.curr_pos_color_scatterplot)


    def on_thresh_change(self, thresh_type, thresh_val):
        if self.apply_thresh:
            if thresh_type == 'lower':
                self.lower_thresh = thresh_val
            elif thresh_type == 'upper':
                self.upper_thresh = thresh_val
            else:
                raise Exception(f'Unknown threshold type given: {thresh_type}')

            self.glvw_color_vis.set_plot(plot=self.curr_img_scatterplot)
            self.glvw_channel_vis.set_plot(plot=self.curr_pos_color_scatterplot)
            self.channel_plot.set_image(self.curr_image_slice)


    def on_apply_crop_toggle(self, should_apply_crop):
        self.apply_crop = should_apply_crop
        self.on_img_modify()


    def on_apply_thresh_toggle(self, should_apply_thresh):
        self.apply_thresh = should_apply_thresh
        self.channel_lower_thresh_slider.setEnabled(self.apply_thresh)
        self.channel_upper_thresh_slider.setEnabled(self.apply_thresh)
        self.on_img_modify()


    def on_apply_cluster_toggle(self, should_apply_cluster):
        self.apply_cluster = should_apply_cluster
        self.cluster_cbox.setEnabled(self.apply_cluster)
        self.cluster_settings_widget.setEnabled(self.apply_cluster)
        self.run_clustering_button.setEnabled(self.apply_cluster)

        self.on_img_modify()


    def on_apply_hist_toggle(self, should_apply_hist):
        self.apply_hist = should_apply_hist
        self.on_img_modify()


    def on_run_clustering(self):
        if self.apply_cluster:
            with GuiBusyLock(self):
                self.clusterer_controller.run_clustering(self.cv_img, self.color_mode)


    def on_img_modify(self):
        if self.apply_crop:
            (x_min, y_min, x_max, y_max) = self.roi_bounds
            self.cropped_img_plot.set_image(self.cv_img.RGB[y_min:y_max, x_min:x_max])
        else:
            self.cropped_img_plot.set_image(self.cv_img.RGB)

        self.glvw_color_vis.set_plot(plot=self.curr_img_scatterplot)
        self.glvw_channel_vis.set_plot(plot=self.curr_pos_color_scatterplot)


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
