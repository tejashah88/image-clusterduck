import numpy as np
import cv2

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from cv_img import CvImg
from image_plotter import ImagePlotter
from plot_3d import Plot3D

# Set window for white blackgroud and black foreground
# pg.setConfigOption('background', 'w')
# pg.setConfigOption('foreground', 'k')

# Interpret image data as row-major instead of col-major
pg.setConfigOptions(imageAxisOrder='row-major')

app = pg.mkQApp()

gl_layout = pg.GraphicsWindow()
gl_layout.setWindowTitle('Image Cluster Analysis')


IMG_FILENAME = './test-images/starry-night.jpg'
MAX_PIXELS = 100 * 1000000
DEFAULY_SCALE_FACTOR = 3

COLOR_MODE = 'RGB'
CHANNEL_INDEX = 0

ALL_COLOR_SPACES = [ 'RGB', 'YUV', 'YCrCb', 'LAB', 'LUV', 'HLS', 'HSV', 'XYZ' ]
COLOR_SPACE_LABELS = {
    'RGB': ('Red', 'Green', 'Blue'),
    'YUV': ('Luma (Brightness)', 'U (blue projection)', 'V (red projection)'),
    'YCrCb': ('Luma (Brightness)', 'Cr (Red from Luma)', 'Cb (Blue from Luma)'),
    'LAB': ('Lightness', 'A (Green to Red)', 'B (Blue to Yellow)'),
    'LUV': ('L (Brightness)', 'U', 'V'),
    'HLS': ('Hue', 'Lightness', 'Saturation'),
    'HSV': ('Hue', 'Saturation', 'Value'),
    'XYZ': ('X', 'Y', 'Z'),
}


def make_img_scatterplot(cv_img, color_mode, scale_factor=DEFAULY_SCALE_FACTOR):
    pos_arr = cv_img[color_mode].reshape(-1, 3) / 255 * scale_factor
    color_arr = cv_img.RGB.reshape(-1, 3) / 255

    return gl.GLScatterPlotItem(
        pos=pos_arr, color=color_arr,
        size=1, pxMode=True,
        glOptions='opaque'
    )


def make_pos_to_color_scatterplot(cv_img, color_mode, ch_index, scale_factor=DEFAULY_SCALE_FACTOR):
    rgb_img = cv_img.RGB
    converted_img = cv_img[color_mode]

    rows, cols = converted_img.shape[:2]
    r_arr, c_arr = np.mgrid[0:rows, 0:cols]
    channel_arr = converted_img[:, :, ch_index]

    color_arr = rgb_img.reshape(-1, 3)
    pos_arr = np.vstack((r_arr.flatten(), c_arr.flatten(), channel_arr.flatten())).T

    pos_arr = pos_arr / np.array([ max(rows, cols), max(rows, cols), max(pos_arr[:, 2]) ])
    pos_arr = pos_arr * np.array([ scale_factor, scale_factor, scale_factor / 2 ])
    color_arr = color_arr / 255

    return gl.GLScatterPlotItem(
        pos=pos_arr, color=color_arr,
        size=1, pxMode=True,
        glOptions='opaque'
    )


def make_pixelbox(num_pixels=1000000, scale_factor=DEFAULY_SCALE_FACTOR):
    pos = np.random.random(size=(num_pixels, 3)) * (2 * scale_factor) - scale_factor
    color = np.random.random(size=(num_pixels, 4))

    return gl.GLScatterPlotItem(
        pos=pos, color=color,
        size=1, pxMode=True,
        glOptions='opaque'
    )


# Link the image plot axes together for consistent panning and zooming
def setup_axes_links(leader_plot, follower_plots):
    for plot in follower_plots:
        plot.setXLink(leader_plot)
        plot.setYLink(leader_plot)


img = cv2.imread(IMG_FILENAME)
if img is None:
    print(f'Error: Unable to load image from {IMG_FILENAME}')
    exit(-1)

height, width = img.shape[:2]
num_pixels = width * height
resize_factor = 1 / ( (num_pixels / MAX_PIXELS) ** 0.5 )
resize_factor = min(resize_factor, 1)
print('Resize factor:', resize_factor)

if resize_factor < 1:
    img = cv2.resize(img, None, fx=resize_factor, fy=resize_factor)

input_img   = CvImg.from_ndarray(img)
input_rgb   = input_img.RGB
input_gray  = input_img.GRAY

print('Original number of pixels:', num_pixels)


orig_img_plot = ImagePlotter(title='Original Image', img=input_rgb, enable_roi=True)
glvw_color_vis = Plot3D(plot=make_img_scatterplot(input_img, COLOR_MODE))

channel_plot = ImagePlotter(title=COLOR_SPACE_LABELS[COLOR_MODE][CHANNEL_INDEX], img=input_img[COLOR_MODE][:, :, CHANNEL_INDEX])
glvw_channel = Plot3D(plot=make_pos_to_color_scatterplot(input_img, COLOR_MODE, CHANNEL_INDEX))

setup_axes_links(orig_img_plot, [channel_plot])

# Setup color space combo box
color_space_cbox = QtGui.QComboBox()
color_space_cbox.addItems(ALL_COLOR_SPACES)

def on_color_space_change(cspace_index):
    global COLOR_MODE
    COLOR_MODE = ALL_COLOR_SPACES[cspace_index]

    glvw_color_vis.set_plot(plot=make_img_scatterplot(input_img, COLOR_MODE))

    channel_plot.setTitle(title=COLOR_SPACE_LABELS[COLOR_MODE][CHANNEL_INDEX])
    channel_plot.set_image(img=input_img[COLOR_MODE][:, :, CHANNEL_INDEX])

    glvw_channel.set_plot(plot=make_pos_to_color_scatterplot(input_img, COLOR_MODE, CHANNEL_INDEX))

    channel_cbox.clear()
    channel_cbox.addItems(COLOR_SPACE_LABELS[COLOR_MODE])
    channel_cbox.setCurrentIndex(-1)
    channel_cbox.setCurrentIndex(CHANNEL_INDEX)

    on_channel_view_change(CHANNEL_INDEX)


color_space_cbox.currentIndexChanged.connect(on_color_space_change)

# Setup channel combo box
channel_cbox = QtGui.QComboBox()
channel_cbox.addItems(COLOR_SPACE_LABELS[COLOR_MODE])

def on_channel_view_change(ch_index):
    global COLOR_MODE
    new_title = COLOR_SPACE_LABELS[COLOR_MODE][ch_index]
    new_img = input_img[COLOR_MODE][:, :, ch_index]

    # Update the title
    channel_plot.setTitle(new_title)

    # Update the image
    channel_plot.set_image(new_img)

    # Update the scatterplot
    new_scatter = make_pos_to_color_scatterplot(input_img, COLOR_MODE, ch_index)
    glvw_channel.plt_item.setData(pos=new_scatter.pos, color=new_scatter.color)

channel_cbox.currentIndexChanged.connect(on_channel_view_change)

# Setup widgets according to given grid layout
grid_layout = QtGui.QGridLayout()

grid_layout.addWidget(color_space_cbox, 0, 0)
grid_layout.addWidget(channel_cbox, 0, 1)

grid_layout.addWidget(orig_img_plot, 1, 0)
grid_layout.addWidget(glvw_color_vis, 2, 0)

grid_layout.addWidget(channel_plot, 1, 1)
grid_layout.addWidget(glvw_channel, 2, 1)

gl_layout.setLayout(grid_layout)


gl_layout.resize(grid_layout.sizeHint() + QtCore.QSize(10, 10))
gl_layout.show()


## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
