import os
import traceback
from concurrent.futures import CancelledError

import numpy as np
import cv2

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl

import sklearn.cluster
from pebble import concurrent
from mss import mss

from src.constants import *
from src.cv_img import CvImg
from src.components.qtrangeslider import QRangeSlider
from src.components.image_plotter import ImagePlotter
from src.components.image_hist_plotter import ImageHistPlotter
from src.components.global_data_tree import GlobalDataTreeWidget
from src.components.plot_3d import Plot3D
from src.gui_busy_lock import GuiBusyLock
from src.image_clusterers import CLUSTER_ALGORITHMS

DEFAULT_IMG_FILENAME = './test-images/starry-night.jpg'

DIALOG_SUPPORTED_IMG_EXTS = ''
for title, exts in SUPPORTED_IMG_EXTS.items():
    exts_str = ' '.join([f'*.{ext}' for ext in exts])
    DIALOG_SUPPORTED_IMG_EXTS += f'{title} ({exts_str});;'
DIALOG_SUPPORTED_IMG_EXTS += 'All Files (*)'

HOME_DIR = os.path.expanduser('~')
HOME_DIR = os.path.curdir # FIXME

DEFAULT_MAX_PIXELS = 10 ** 6

# NOTE: These constants will be initialized later
SCREEN_WIDTH = -1
SCREEN_HEIGHT = -1

ALL_CLUSTER_ALGORITHMS = list(CLUSTER_ALGORITHMS.keys())
IMG_CLUSTERERS = list(CLUSTER_ALGORITHMS.values())

CLUSTER_INPUTS = {
    'color'  : 'Color-only',
    'spatial': 'Spatial-only',
    'both'   : 'Color & Spatial',
}

INTERNAL_CLUSTER_INPUTS = list(CLUSTER_INPUTS.keys())
CLUSTER_INPUT_TYPES = list(CLUSTER_INPUTS.values())


IMG_SCPLOT_SCALE = 4
CH_SCPLOT_SCALE = 5
CH_SCPLOT_SCALE_Z = 2
CH_PLOT_GRID_SZ = 8

def process_img_plot_mouse_event(img_plot, curr_img, fn):
    def handle_mouse_event(mouse_pos):
        if img_plot.sceneBoundingRect().contains(mouse_pos):
            mouse_point = img_plot.getViewBox().mapSceneToView(mouse_pos)
            (mouse_x, mouse_y) = int(mouse_point.x()), int(mouse_point.y())
            (height, width) = curr_img.shape[:2]
            if (0 <= mouse_y and mouse_y < height) and (0 <= mouse_x and mouse_x < width):
                return fn(mouse_x, mouse_y, curr_img[mouse_y, mouse_x])
    return handle_mouse_event


def cluster_points_plot(color_centers, rgb_colored_centers, scale_factor=IMG_SCPLOT_SCALE):
    return gl.GLScatterPlotItem(
        pos=color_centers / 255 * scale_factor, color=rgb_colored_centers / 255,
        size=0.75, pxMode=not True,
        glOptions='opaque'
    )


def img_scatterplot(cv_img, color_mode, crop_bounds=None, thresh_bounds=None, scale_factor=IMG_SCPLOT_SCALE):
    rgb_img = cv_img.RGB
    converted_img = cv_img[color_mode]

    if crop_bounds is not None:
        x_min, y_min, x_max, y_max = crop_bounds
    else:
        height, width = rgb_img.shape[:2]
        x_min, y_min, x_max, y_max = (0, 0, width, height)

    rgb_img = rgb_img[y_min:y_max, x_min:x_max]
    converted_img = converted_img[y_min:y_max, x_min:x_max]

    if thresh_bounds is None:
        thresh_bounds = [(0, 255), (0, 255), (0, 255)]

    for (ch_index, bounds) in enumerate(thresh_bounds):
        lower_ch, upper_ch = bounds
        channel_arr = converted_img[:, :, ch_index]

        thresh_indicies = ( (channel_arr < lower_ch) | (channel_arr > upper_ch) )
        converted_img[thresh_indicies] = -1

    pos_arr = converted_img.reshape(-1, 3)
    color_arr = rgb_img.reshape(-1, 3) / 255

    non_zero_pixels = np.all(pos_arr != -1, axis=1)
    pos_arr = pos_arr[non_zero_pixels]
    color_arr = color_arr[non_zero_pixels]

    pos_arr = converted_img.reshape(-1, 3) / 255 * scale_factor

    return gl.GLScatterPlotItem(
        pos=pos_arr, color=color_arr,
        size=1, pxMode=True,
        glOptions='opaque'
    )


def pos_color_scatterplot(cv_img, color_mode, ch_index, crop_bounds=None, thresh_bounds=None, scale_factor=CH_SCPLOT_SCALE, scale_z=CH_SCPLOT_SCALE_Z):
    rgb_img = cv_img.RGB.copy()
    converted_img = cv_img[color_mode].copy()

    if crop_bounds is not None:
        x_min, y_min, x_max, y_max = crop_bounds
    else:
        height, width = rgb_img.shape[:2]
        x_min, y_min, x_max, y_max = (0, 0, width, height)

    rgb_img = rgb_img[y_min:y_max, x_min:x_max]
    converted_img = converted_img[y_min:y_max, x_min:x_max]

    if thresh_bounds is not None:
        lower_ch, upper_ch = thresh_bounds[ch_index]
    else:
        lower_ch, upper_ch = (0, 255)

    rows, cols = converted_img.shape[:2]
    c_arr, r_arr = np.meshgrid(np.arange(cols), np.arange(rows))
    channel_arr = converted_img[:, :, ch_index]

    keep_indicies = ( (channel_arr > lower_ch) & (channel_arr < upper_ch) )
    flat_keep_indices = keep_indicies.flatten()

    flat_r_arr = r_arr.flatten()[flat_keep_indices]
    flat_c_arr = c_arr.flatten()[flat_keep_indices]
    flat_channel_arr = channel_arr.flatten()[flat_keep_indices]

    scaled_dim = scale_factor / max(rows, cols)
    scaled_z = scale_z / 255

    flat_r_arr = (flat_r_arr - rows // 2) * scaled_dim
    flat_c_arr = (flat_c_arr - cols // 2) * scaled_dim
    flat_channel_arr = flat_channel_arr * scaled_z

    pos_arr = np.vstack( (flat_r_arr, flat_c_arr, flat_channel_arr) ).T

    color_arr = rgb_img.reshape(-1, 3)  / 255
    color_arr = color_arr[flat_keep_indices, :]

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


# Load image with approximate max number of pixels
def load_image_max_pixels(input_img, max_pixels):
    num_pixels = image_num_pixels(input_img)
    if num_pixels > max_pixels:
        resize_factor = img_resize_factor(input_img, max_pixels)
        resized_img = cv2.resize(input_img, None, fx=resize_factor, fy=resize_factor)
    else:
        resize_factor = 1
        resized_img = input_img[:, :, :]

    return resized_img


# Returns the number of pixels in a 2D or 3D image
def image_num_pixels(img):
    return int(np.product(img.shape[:2]))

# Return required resize factor to shrink image to contain given max number of pixels
def img_resize_factor(input_img, max_pixels):
    resize_factor = 1 / ( (image_num_pixels(input_img) / max_pixels) ** 0.5 )
    if resize_factor < 1:
        return resize_factor
    return 1


# Interpret image data as row-major instead of col-major
pg.setConfigOptions(imageAxisOrder='row-major')


class MyWindow(pg.GraphicsLayoutWidget):
    def __init__(self):
        super().__init__()

        self.input_img = None
        self.cv_img = None

        self.dataset_mode = False
        self.dataset_imgs = []
        self.dataset_index = None

        self.ch_index = 0
        self.cs_index = 0
        self.cluster_algo_index = 0
        self.cluster_input_index = 0

        self.orig_img_plot = None
        self.glvw_color_vis = None
        self.channel_plot = None
        self.glvw_channel_vis = None

        self.roi = None
        self.menubar = None
        self.statusbar = None

        self.apply_crop = False
        self.apply_thresh = False
        self.mod_img_realtime = False

        self.max_pixels_to_load = DEFAULT_MAX_PIXELS
        self.channel_thresholds = [(0, 255), (0, 255), (0, 255)]

        self.cluster_future = None
        self.cluster_check_timer = None

        self.main_window = None


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
    def cluster_input_mode(self):
        return INTERNAL_CLUSTER_INPUTS[self.cluster_input_index]


    @property
    def curr_image(self):
        return self.cv_img[self.color_mode]


    @property
    def curr_image_gray(self):
        return self.cv_img.GRAY


    @property
    def curr_image_cropped(self):
        if self.apply_crop:
            x_min, y_min, x_max, y_max = self.roi_bounds
            return self.curr_image[y_min:y_max, x_min:x_max]
        else:
            return self.curr_image


    @property
    def curr_image_gray_cropped(self):
        if self.apply_crop:
            x_min, y_min, x_max, y_max = self.roi_bounds
            return self.curr_image_gray[y_min:y_max, x_min:x_max]
        else:
            return self.curr_image_gray


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
        height, width = self.cv_img.RGB.shape[:2]
        if self.apply_crop:
            x, y, w, h = self.roi.parentBounds().toAlignedRect().getRect()
            x_min, y_min = max(x, 0), max(y, 0)
            x_max, y_max = min(x + w, width), min(y + h, height)
            return (x_min, y_min, x_max, y_max)
        else:
            return (0, 0, width, height)


    @property
    def thresh_bounds(self):
        if self.apply_thresh:
            return self.channel_thresholds[self.ch_index]
        return None


    @property
    def curr_img_scatterplot(self):
        return img_scatterplot(
            self.cv_img, self.color_mode,
            crop_bounds=self.roi_bounds,
            thresh_bounds=self.channel_thresholds if self.apply_thresh else None
        )


    @property
    def curr_pos_color_scatterplot(self):
        return pos_color_scatterplot(
            self.cv_img, self.color_mode, self.ch_index,
            crop_bounds=self.roi_bounds,
            thresh_bounds=self.channel_thresholds if self.apply_thresh else None
        )


    def load_image_file(self, img_path, max_pixels):
        input_img = cv2.imread(img_path)
        if input_img is None:
            QtGui.QMessageBox.warning(self, 'Error!', f'Unable to load image from "{img_path}"')

            if self.gui_ready:
                return
            else:
                exit(-1)

        self.load_image(input_img, max_pixels)
        self.set_window_title(f'Now viewing "{img_path.split("/")[-1]}"')


    def load_image(self, input_img, max_pixels):
        if max_pixels is None:
            max_pixels = self.max_pixels_to_load

        with GuiBusyLock(self):
            self.input_img = input_img
            resized_img = load_image_max_pixels(self.input_img, max_pixels)
            self.cv_img = CvImg.from_ndarray(resized_img)

            if self.gui_ready:
                self.data_tree['Image Info/Total Pixels'] = image_num_pixels(self.input_img)
                self.data_tree['Image Info/Pixels Loaded'] = image_num_pixels(self.curr_image)
                self.data_tree['Image Info/Resize Factor'] = img_resize_factor(self.input_img, max_pixels)
                self.data_tree['Image Info/Original Image Size'] = np.array(self.input_img.shape[:2][::-1])
                self.data_tree['Image Info/Loaded Image Size'] = np.array(self.curr_image.shape[:2][::-1])

                self.orig_img_plot.set_image(self.cv_img.RGB)
                self.on_color_space_change(self.cs_index)


    def setup_gui(self):
        if self.cv_img is None:
            raise Exception('Error: Image has not been loaded yet! Please load an image before calling setup_gui()')

        # Setup widgets according to grid layout
        self.main_grid_layout = QtGui.QGridLayout()

        # Optimal plot size is determined so that the app takes 75% total width and 80% total height (for 2 plots high and 3 plots wide)
        optimal_plot_size = (SCREEN_WIDTH // 4, SCREEN_HEIGHT // 2.5)

        # Setup main plots
        self.orig_img_plot = ImagePlotter(title='Original Image', img=self.cv_img.RGB, enable_crosshair=True, size=optimal_plot_size)
        self.glvw_color_vis = Plot3D(plot=self.curr_img_scatterplot, size=optimal_plot_size)

        self.channel_plot = ImagePlotter(title=self.channel_mode, img=self.curr_image_slice, size=optimal_plot_size)
        self.glvw_channel_vis = Plot3D(plot=self.curr_pos_color_scatterplot, enable_axes=False, size=optimal_plot_size)
        self.glvw_channel_vis.grid_item.setPosition(x=-CH_PLOT_GRID_SZ / 2, y=-CH_PLOT_GRID_SZ / 2, z=0)
        self.glvw_channel_vis.grid_item.setSize(x=CH_PLOT_GRID_SZ, y=CH_PLOT_GRID_SZ, z=0)

        # Tie the axes bewteen the original image plot and the channel sliced image plot
        setup_axes_links(self.orig_img_plot, [self.channel_plot])

        # Layout main plots
        self.main_grid_layout.addWidget(self.orig_img_plot, 0, 0)
        self.main_grid_layout.addWidget(self.glvw_color_vis, 1, 0)

        self.main_grid_layout.addWidget(self.channel_plot, 0, 1)
        self.main_grid_layout.addWidget(self.glvw_channel_vis, 1, 1)

        # Setup the color histogram plot
        self.color_hist_plot = ImageHistPlotter(title='Color/Gray Histogram', size=optimal_plot_size)
        self.color_hist_plot.plot_hist(self.curr_image_cropped, self.curr_image_gray_cropped)
        self.main_grid_layout.addWidget(self.color_hist_plot, 0, 2)

        # Setup settings/data tabs
        info_tabs = QtGui.QTabWidget()
        general_data_settings_tab = QtGui.QWidget()
        cluster_settings_tab = QtGui.QWidget()

        info_tabs.addTab(general_data_settings_tab, 'Settings/Data')
        info_tabs.addTab(cluster_settings_tab, 'Clustering')


        # Lay everything out for general settings/data tab
        self.general_settings_layout = QtGui.QGridLayout()

        # Setup max pixels loading slider
        self.max_pixels_slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.max_pixels_slider.setMinimum(0)
        self.max_pixels_slider.setMaximum(10)
        self.max_pixels_slider.setValue(6)
        self.max_pixels_slider.setTickPosition(QtGui.QSlider.TicksBelow)
        self.max_pixels_slider.setTickInterval(1)

        def on_max_pixels_slider_change(val):
            self.max_pixels_to_load = 10 ** val
            self.load_image(self.input_img, self.max_pixels_to_load)
            self.data_tree['Image Info/Pixels Loaded'] = image_num_pixels(self.curr_image)

        self.max_pixels_slider.valueChanged.connect(on_max_pixels_slider_change)

        self.general_settings_layout.addWidget(QtGui.QLabel('Max Pixels (10^x):'), 0, 0)
        self.general_settings_layout.addWidget(self.max_pixels_slider, 0, 1)

        # Setup image realtime modding check box
        self.mod_img_realtime_box = QtGui.QCheckBox()
        self.mod_img_realtime_box.setChecked(self.mod_img_realtime)
        self.mod_img_realtime_box.toggled.connect(self.on_mod_img_realtime_toggle)
        self.general_settings_layout.addWidget(QtGui.QLabel('Realtime updates:'), 1, 0)
        self.general_settings_layout.addWidget(self.mod_img_realtime_box, 1, 1)


        # Setup color space combo box
        self.color_space_cbox = QtGui.QComboBox()
        self.color_space_cbox.addItems(ALL_COLOR_SPACES)
        self.color_space_cbox.setCurrentIndex(self.cs_index)
        self.color_space_cbox.currentIndexChanged.connect(self.on_color_space_change)

        self.general_settings_layout.addWidget(QtGui.QLabel('Color Space:'), 2, 0)
        self.general_settings_layout.addWidget(self.color_space_cbox, 2, 1)

        # Setup channel combo box
        self.channel_cbox = QtGui.QComboBox()
        self.channel_cbox.addItems(COLOR_SPACE_LABELS[self.color_mode])
        self.channel_cbox.setCurrentIndex(self.ch_index)
        self.channel_cbox.currentIndexChanged.connect(self.on_channel_view_change)

        self.general_settings_layout.addWidget(QtGui.QLabel('Channel:'), 3, 0)
        self.general_settings_layout.addWidget(self.channel_cbox, 3, 1)

        # Setup cropping checkbox
        self.apply_crop_box = QtGui.QCheckBox()
        self.apply_crop_box.setChecked(self.apply_crop)
        self.apply_crop_box.toggled.connect(self.on_apply_crop_toggle)
        self.general_settings_layout.addWidget(QtGui.QLabel('Apply Cropping:'), 4, 0)
        self.general_settings_layout.addWidget(self.apply_crop_box, 4, 1)

        # Setup thresholding checkboxes
        self.apply_thresh_box = QtGui.QCheckBox()
        self.apply_thresh_box.setChecked(self.apply_thresh)
        self.apply_thresh_box.toggled.connect(self.on_apply_thresh_toggle)
        self.general_settings_layout.addWidget(QtGui.QLabel('Apply Thresholding:'), 5, 0)
        self.general_settings_layout.addWidget(self.apply_thresh_box, 5, 1)

        # Setup thresholding sliders for all channels
        thresh_row_offset = 6
        self.all_channel_thresh_sliders = []
        self.all_channel_labels = []

        for i in range(3):
            # Setup thresholding channel label
            channel_label = QtGui.QLabel(f'Threshold ({COLOR_SPACE_LABELS[self.color_mode][i]}):')
            self.general_settings_layout.addWidget(channel_label, thresh_row_offset + i, 0)
            self.all_channel_labels += [channel_label]

            # Setup thresholding channel range slider
            channel_thresh_slider = QRangeSlider(QtCore.Qt.Horizontal)
            channel_thresh_slider.range = (0, 255)
            channel_thresh_slider.values = (0, 255)
            channel_thresh_slider.setEnabled(False)
            self.general_settings_layout.addWidget(channel_thresh_slider, thresh_row_offset + i, 1)
            self.all_channel_thresh_sliders += [channel_thresh_slider]

        # Setup the data tree widget
        # NOTE: Top level keys will be rendered in reverse insertion order
        initial_data = {
            'Image Controls': {
                'Crop Dimensions': np.array(self.roi_bounds),
                'Channel Thresholds': np.array(self.channel_thresholds).T
            },
            'Mouse Info': {
                'Mouse Location': np.array([-1, -1]),
                'Color at Mouse': np.array([-1, -1, -1]),
            },
            'Image Info': {
                'Total Pixels': image_num_pixels(self.input_img),
                'Pixels Loaded': image_num_pixels(self.curr_image),
                'Resize Factor': img_resize_factor(self.input_img, self.max_pixels_to_load),
                'Original Image Size': np.array(self.input_img.shape[:2][::-1]),
                'Loaded Image Size': np.array(self.curr_image.shape[:2][::-1]),
            },
        }

        self.data_tree = GlobalDataTreeWidget()
        self.data_tree.set_data(initial_data)

        self.general_settings_layout.addWidget(self.data_tree, 9, 0, 1, 2)


        def handle_on_mouse_hover(x, y, color):
            self.data_tree['Mouse Info/Mouse Location'] = np.array([x, y])
            self.data_tree['Mouse Info/Color at Mouse'] = color

        show_color_on_hover = process_img_plot_mouse_event(self.orig_img_plot, self.curr_image, handle_on_mouse_hover)
        self.orig_img_plot.scene().sigMouseMoved.connect(show_color_on_hover)


        # HACK: Add dummy label widget to squish all widgets to the top
        self.general_settings_layout.addWidget(QtGui.QLabel(''), 10, 0, 999, 2)

        # Place all general settings widgets in 'Settings' tab
        general_data_settings_tab.setLayout(self.general_settings_layout)


        # Lay everything out for clustering settings tab
        self.clustering_settings_layout = QtGui.QGridLayout()

        # Setup clustering algorithm combo box
        self.cluster_algo_cbox = QtGui.QComboBox()
        self.cluster_algo_cbox.addItems(ALL_CLUSTER_ALGORITHMS)
        self.cluster_algo_cbox.setCurrentIndex(self.cluster_algo_index)
        self.cluster_algo_cbox.currentIndexChanged.connect(self.on_cluster_algo_change)

        self.clustering_settings_layout.addWidget(QtGui.QLabel('Cluster Algorithm:'), 0, 0)
        self.clustering_settings_layout.addWidget(self.cluster_algo_cbox, 0, 1)

        # Setup clustering algorithm input data combo box
        self.cluster_input_cbox = QtGui.QComboBox()
        self.cluster_input_cbox.addItems(CLUSTER_INPUT_TYPES)
        self.cluster_input_cbox.setCurrentIndex(self.cluster_input_index)
        self.cluster_input_cbox.currentIndexChanged.connect(self.on_cluster_input_change)

        self.clustering_settings_layout.addWidget(QtGui.QLabel('Cluster Input Type:'), 1, 0)
        self.clustering_settings_layout.addWidget(self.cluster_input_cbox, 1, 1)

        # Setup the cluster sub-settings widgets
        self.clusterer_controller = IMG_CLUSTERERS[self.cluster_algo_index]
        cluster_sub_settings_layout = self.clusterer_controller.setup_settings_layout()

        self.cluster_settings_widget = QtGui.QWidget()
        self.cluster_settings_widget.setLayout(cluster_sub_settings_layout)
        self.clustering_settings_layout.addWidget(self.cluster_settings_widget, 2, 0, 1, 2)

        # Setup clustering buttons
        self.run_clustering_button = QtGui.QPushButton('Run Clustering')
        self.run_clustering_button.clicked.connect(self.on_run_clustering)
        self.run_clustering_button.setEnabled(True)
        self.clustering_settings_layout.addWidget(self.run_clustering_button, 3, 0)

        self.cancel_clustering_button = QtGui.QPushButton('Cancel Clustering')
        self.cancel_clustering_button.clicked.connect(self.on_cancel_clustering)
        self.cancel_clustering_button.setEnabled(False)
        self.clustering_settings_layout.addWidget(self.cancel_clustering_button, 3, 1)

        # HACK: Add dummy label widget to squish all widgets to the top
        self.clustering_settings_layout.addWidget(QtGui.QLabel(''), 4, 0, 999, 2)

        # Place all cluster settings widgets in 'Clustering' tab
        cluster_settings_tab.setLayout(self.clustering_settings_layout)

        # Add the tabs into the main layout
        self.main_grid_layout.addWidget(info_tabs, 1, 2)

        # Set the layout and resize the window accordingly
        self.setLayout(self.main_grid_layout)
        self.resize(self.main_grid_layout.sizeHint() + QtCore.QSize(10, 30))


    def bind_to_main_window(self, main_window):
        self.main_window = main_window
        self.main_window.setCentralWidget(self)

        self.setup_menubar(self.main_window)
        self.setup_statusbar(self.main_window)
        self.setup_shortcuts()

        self.autosize()


    def open_file_dialog(self, title, supported_exts, starting_dir=HOME_DIR):
        filename, _ = pg.FileDialog().getOpenFileName(self, title, starting_dir, supported_exts)
        return filename


    def save_file_dialog(self, title, supported_exts, starting_dir=HOME_DIR):
        filename, _ = pg.FileDialog().getSaveFileName(self, title, starting_dir, supported_exts)
        return filename


    def open_folder_dialog(self, title, starting_dir=HOME_DIR):
        dirname = pg.FileDialog().getExistingDirectory(self, title, starting_dir)
        return dirname


    def on_color_space_change(self, cspace_index):
        with GuiBusyLock(self):
            self.cs_index = cspace_index

            # NOTE: Temporarily disable the 'currentIndexChanged' since
            # it'll be triggered when removing and adding new items
            self.channel_cbox.currentIndexChanged.disconnect()
            self.channel_cbox.clear()
            self.channel_cbox.addItems(COLOR_SPACE_LABELS[self.color_mode])
            self.channel_cbox.currentIndexChanged.connect(self.on_channel_view_change)

            for i in range(3):
                channel_label = self.all_channel_labels[i]
                channel_label.setText(f'Threshold ({COLOR_SPACE_LABELS[self.color_mode][i]}):')

                channel_thresh_slider = self.all_channel_thresh_sliders[i]
                self.channel_thresholds[i] = (0, 255)
                channel_thresh_slider.values = (0, 255)

            self.channel_plot.setTitle(title=self.channel_mode)

            self.update_all_plots()
            self.channel_plot.autoRange()
            self.glvw_color_vis.remove_cluster_plot()


    def on_channel_view_change(self, ch_index):
        with GuiBusyLock(self):
            self.ch_index = ch_index

            self.channel_plot.setTitle(title=self.channel_mode)
            self.update_all_plots()
            self.channel_plot.autoRange()


    def on_cluster_algo_change(self, cluster_algo_index):
        self.cluster_algo_index = cluster_algo_index

        self.clusterer_controller = IMG_CLUSTERERS[self.cluster_algo_index]
        cluster_settings_layout = self.clusterer_controller.setup_settings_layout()

        old_widget = self.cluster_settings_widget
        self.cluster_settings_widget = QtGui.QWidget()
        self.cluster_settings_widget.setLayout(cluster_settings_layout)

        self.clustering_settings_layout.replaceWidget(old_widget, self.cluster_settings_widget)
        QtCore.QObjectCleanupHandler().add(old_widget)
        self.clustering_settings_layout.update()


    def on_cluster_input_change(self, cluster_input_index):
        self.cluster_input_index = cluster_input_index


    def on_crop_modify(self):
        if self.apply_crop:
            self.update_all_plots()


    def on_crop_modify_realtime(self):
        if self.apply_crop:
            self.data_tree['Image Controls/Crop Dimensions'] = np.array(self.roi_bounds)
            self.update_2d_plots()

            if self.mod_img_realtime:
                self.update_3d_plots()


    def on_thresh_change(self, thresh_ch_index, lower_val, upper_val):
        if self.apply_thresh:
            self.channel_thresholds[thresh_ch_index] = (lower_val, upper_val)
            self.update_all_plots()


    def on_thresh_change_realtime(self, thresh_ch_index, lower_val, upper_val):
        if self.apply_thresh:
            self.channel_thresholds[thresh_ch_index] = (lower_val, upper_val)
            self.data_tree['Image Controls/Channel Thresholds'] = np.array(self.channel_thresholds).T
            self.update_2d_plots()

            if self.mod_img_realtime:
                self.update_3d_plots()


    def on_apply_crop_toggle(self, should_apply_crop):
        self.apply_crop = should_apply_crop

        if self.apply_crop:
            self.orig_img_plot.enable_roi_rect()
            self.roi = self.orig_img_plot.roi_item
            self.roi.sigRegionChanged.connect(self.on_crop_modify_realtime)
            self.roi.sigRegionChangeFinished.connect(self.on_crop_modify)
        else:
            self.roi.sigRegionChanged.disconnect()
            self.roi.sigRegionChangeFinished.disconnect()
            self.roi = None
            self.orig_img_plot.disable_roi_rect()

        self.data_tree['Image Controls/Crop Dimensions'] = np.array(self.roi_bounds)
        self.update_all_plots()


    def on_mod_img_realtime_toggle(self, should_mod_img_realtime):
        self.mod_img_realtime = should_mod_img_realtime


    def on_apply_thresh_toggle(self, should_apply_thresh):
        self.apply_thresh = should_apply_thresh
        for (i, channel_thresh_slider) in enumerate(self.all_channel_thresh_sliders):
            channel_thresh_slider.setEnabled(self.apply_thresh)

            channel_thresh_value_changed_realtime = lambda i: (lambda lower, upper: self.on_thresh_change_realtime(i, lower, upper))
            channel_thresh_value_changed = lambda i: (lambda lower, upper: self.on_thresh_change(i, lower, upper))

            if self.apply_thresh:
                channel_thresh_slider.valueChanged.connect(channel_thresh_value_changed_realtime(i))
                channel_thresh_slider.valueChangedFinished.connect(channel_thresh_value_changed(i))
            else:
                channel_thresh_slider.valueChanged.disconnect()
                channel_thresh_slider.valueChangedFinished.disconnect()

        self.update_all_plots()


    @property
    def is_clustering(self):
        return self.cluster_future is not None and self.cluster_future.running()


    def on_run_clustering(self):
        if not self.is_clustering:
            self.run_clustering_button.setEnabled(False)
            self.cancel_clustering_button.setEnabled(True)
            self.glvw_color_vis.remove_cluster_plot()


            @concurrent.process
            def _run_clustering(cv_img, color_mode, input_mode, roi_bounds):
                outcome = {
                    'results': None,
                    'exception': None,
                }

                try:
                    results = self.clusterer_controller.run_clustering(cv_img, color_mode, input_mode, roi_bounds)
                    color_centers, color_labels, rgb_colored_centers, cluster_error, num_iterations = results

                    outcome['results'] = (color_centers, rgb_colored_centers)
                except Exception as ex:
                    err_name = str(ex)
                    err_type = str(type(ex))
                    err_stacktrace = ''.join(traceback.format_tb(ex.__traceback__))

                    outcome['exception'] = {
                        'name': err_name,
                        'type': err_type,
                        'stacktrace': err_stacktrace,
                    }

                return outcome


            def _check_clustering_results():
                if self.cluster_future.done():
                    self.cluster_check_timer.stop()

                    try:
                        outcome = self.cluster_future.result()

                        if outcome['exception'] is not None:
                            error_msg = f'A problem occurred when running the clustering algorithm:'
                            error_msg += f"\n{outcome['exception']['name']}"
                            error_msg += f"\n{outcome['exception']['stacktrace']}"
                            QtGui.QMessageBox.warning(self, 'Error!', error_msg)
                        else:
                            color_centers, rgb_colored_centers = outcome['results']
                            self.glvw_color_vis.set_cluster_plot(cluster_points_plot(color_centers, rgb_colored_centers))
                    except CancelledError as ex:
                        # NOTE: The user requested to cancel the clustering operation
                        pass
                    finally:
                        self.run_clustering_button.setEnabled(True)
                        self.cancel_clustering_button.setEnabled(False)


            self.cluster_future = _run_clustering(self.cv_img, self.color_mode, self.cluster_input_mode, self.roi_bounds)
            self.cluster_check_timer = QtCore.QTimer()
            self.cluster_check_timer.timeout.connect(_check_clustering_results)
            self.cluster_check_timer.start(250)


    def on_cancel_clustering(self):
        if self.is_clustering:
            self.cluster_future.cancel()

            self.glvw_color_vis.remove_cluster_plot()
            self.run_clustering_button.setEnabled(True)
            self.cancel_clustering_button.setEnabled(False)


    def update_2d_plots(self):
        self.channel_plot.set_image(self.curr_image_slice, auto_range=False)
        self.color_hist_plot.plot_hist(self.curr_image_cropped, self.curr_image_gray_cropped)


    def update_3d_plots(self):
        self.glvw_color_vis.set_plot(plot=self.curr_img_scatterplot)
        self.glvw_channel_vis.set_plot(plot=self.curr_pos_color_scatterplot)


    def update_all_plots(self):
        self.update_2d_plots()
        self.update_3d_plots()


    def setup_menubar(self, main_window):
        self.menubar = main_window.menuBar()
        file_menu = self.menubar.addMenu('File')
        help_menu = self.menubar.addMenu('Help')


        open_image_action = QtGui.QAction('Open Image', self)
        open_image_action.setShortcut('Ctrl+O')
        open_image_action.setStatusTip('Open Image')

        def on_img_file_select():
            img_path = self.open_file_dialog('Open image file', DIALOG_SUPPORTED_IMG_EXTS)
            if len(img_path) > 0:
                self.dataset_mode = False
                self.dataset_imgs = []
                self.dataset_index = None
                self.load_image_file(img_path, self.max_pixels_to_load)

        open_image_action.triggered.connect(on_img_file_select)
        file_menu.addAction(open_image_action)


        open_dataset_action = QtGui.QAction('Open Dataset', self)
        open_dataset_action.setShortcut('Ctrl+Shift+O')
        open_dataset_action.setStatusTip('Open dataset of images')

        def on_dataset_folder_select():
            dataset_dir = self.open_folder_dialog('Open image dataset folder')
            if len(dataset_dir) > 0:
                raw_paths = [os.path.join(dataset_dir, filepath) for filepath in os.listdir(dataset_dir)]
                dataset_image_paths = [filepath for filepath in raw_paths if os.path.isfile(filepath) and filepath.endswith(ALL_SUPPORTED_IMG_EXTS)]

                self.dataset_mode = True
                self.dataset_imgs = dataset_image_paths
                self.dataset_index = 0
                self.load_image_file(self.dataset_imgs[self.dataset_index], self.max_pixels_to_load)

        open_dataset_action.triggered.connect(on_dataset_folder_select)
        file_menu.addAction(open_dataset_action)


        export_screenshot_action = QtGui.QAction('Export Screenshot', self)
        export_screenshot_action.setShortcut('Ctrl+E')
        export_screenshot_action.setStatusTip('Export screenshot of app')

        def on_export_screenshot_request():
            self.main_window.move(10, 10)

            win_geometry = self.geometry()
            position = self.mapToGlobal(self.geometry().topLeft())
            size = self.geometry().size()

            x, y = position.x(), position.y()
            width, height = size.width(), size.height()

            window_bounds = {
                'top': y - 20,
                'left': x,
                'width': width,
                'height': height,
            }

            with mss() as sct:
                window_view = np.array(sct.grab(window_bounds))
                window_view = cv2.cvtColor(window_view, cv2.COLOR_RGBA2RGB)

            save_filepath = self.save_file_dialog('Save screenshot export', DIALOG_SUPPORTED_IMG_EXTS)
            cv2.imwrite(save_filepath, window_view)

        export_screenshot_action.triggered.connect(on_export_screenshot_request)
        file_menu.addAction(export_screenshot_action)


        exit_action = QtGui.QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(main_window.close)
        file_menu.addAction(exit_action)


    def setup_shortcuts(self):
        QtGui.QShortcut(QtCore.Qt.Key_Left, self, self.load_previous_image_in_dataset)
        QtGui.QShortcut(QtCore.Qt.Key_Right, self, self.load_next_image_in_dataset)


    def load_previous_image_in_dataset(self):
        if self.dataset_mode:
            self.dataset_index -= 1
            if self.dataset_index < 0:
                self.dataset_index += len(self.dataset_imgs)
            self.load_image_file(self.dataset_imgs[self.dataset_index], self.max_pixels_to_load)


    def load_next_image_in_dataset(self):
        if self.dataset_mode:
            self.dataset_index += 1
            self.dataset_index %= len(self.dataset_imgs)
            self.load_image_file(self.dataset_imgs[self.dataset_index], self.max_pixels_to_load)


    def setup_statusbar(self, main_window):
        self.statusbar = main_window.statusBar()


    def show_status(self, text, timeout=0):
        if self.statusbar is not None:
            self.statusbar.showMessage(text, timeout)


    def set_window_title(self, text):
        if self.main_window is not None:
            self.main_window.setWindowTitle(text)

    def autosize(self):
        self.main_window.resize(self.size())


## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        app = pg.mkQApp()

        screen_resolution = app.desktop().screenGeometry()
        SCREEN_WIDTH, SCREEN_HEIGHT = screen_resolution.width(), screen_resolution.height()

        with open('src/app.css') as fp:
            app.setStyleSheet('\n'.join(fp.readlines()).strip())

        MainWindow = QtGui.QMainWindow()
        gui = MyWindow()
        gui.load_image_file(DEFAULT_IMG_FILENAME, DEFAULT_MAX_PIXELS)
        gui.setup_gui()
        gui.bind_to_main_window(MainWindow)
        gui.set_window_title(f'Now viewing "{DEFAULT_IMG_FILENAME.split("/")[-1]}"')
        MainWindow.show()

        # HACK: This dummy timer lets us properly Ctrl+C from the app
        timer = QtCore.QTimer()
        timer.timeout.connect(lambda: None)
        timer.start(100)

        sys.exit(app.exec_())
