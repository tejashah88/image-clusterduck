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

main_area = pg.QtGui.QScrollArea()
main_area.setWindowTitle('Image Cluster Analysis')

win = pg.GraphicsLayoutWidget()

IMG_FILENAME = './test-images/bad_cluster_1.png'
COLOR_MODE = 'RGB'
MAX_PIXELS = 100 * 1000000
DEFAULT_INDEX = 0

ALL_COLOR_SPACES = {
    'RGB': ('Red', 'Green', 'Blue'),
    'YUV': ('Luma (Brightness)', 'U (blue projection)', 'V (red projection)'),
    'YCrCb': ('Luma (Brightness)', 'Cr (Red from Luma)', 'Cb (Blue from Luma)'),
    'LAB': ('Lightness', 'A (Green to Red)', 'B (Blue to Yellow)'),
    'LUV': ('L (Brightness)', 'U', 'V'),
    'HLS': ('Hue', 'Lightness', 'Saturation'),
    'HSV': ('Hue', 'Saturation', 'Value'),
    'XYZ': ('X', 'Y', 'Z'),
}


def make_img_scatterplot(cv_img, COLOR_MODE, scale_factor=3):
    pos_arr = cv_img[COLOR_MODE].reshape(-1, 3) / 255 * scale_factor
    color_arr = cv_img.RGB.reshape(-1, 3) / 255

    return gl.GLScatterPlotItem(
        pos=pos_arr, color=color_arr,
        size=1, pxMode=True,
        glOptions='opaque'
    )


def make_pos_to_color_scatterplot(cv_img, COLOR_MODE, ch_index, scale_factor=3):
    rgb_img = cv_img.RGB
    converted_img = cv_img[COLOR_MODE]

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


def make_pixelbox(num_pixels=1000000, scale_factor=3):
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
input_mod   = input_img[COLOR_MODE]

print('Original number of pixels:', num_pixels)

# # Isocurve drawing
# iso = pg.IsocurveItem(level=0.8, pen='g')
# iso.setParentItem(img)
# iso.setZValue(5)

# # Contrast/color control
# hist = pg.HistogramLUTItem()
# hist.setImageItem(img)
# win.addItem(hist)

# # Draggable line for setting isocurve level
# isoLine = pg.InfiniteLine(angle=0, movable=True, pen='g')
# hist.vb.addItem(isoLine)
# hist.vb.setMouseEnabled(y=False) # makes user interaction a little easier
# isoLine.setValue(0.8)
# isoLine.setZValue(1000) # bring iso line above contrast controls

# # Another plot area for displaying ROI data
# win.nextRow()
# p2 = win.addPlot(colspan=2)
# p2.setMaximumHeight(250)


orig_img_plot = ImagePlotter(title='Original Image', img=input_rgb, enable_roi=True)
channel_plot = ImagePlotter(title=ALL_COLOR_SPACES[COLOR_MODE][DEFAULT_INDEX], img=input_mod[:, :, DEFAULT_INDEX])

glvw_color_vis = Plot3D(plot=make_img_scatterplot(input_img, COLOR_MODE))
glvw_channel = Plot3D(plot=make_pos_to_color_scatterplot(input_img, COLOR_MODE, DEFAULT_INDEX))

setup_axes_links(orig_img_plot, [channel_plot])

# Setup channel combo box
channel_cbox = QtGui.QComboBox()
channel_cbox.addItems(ALL_COLOR_SPACES[COLOR_MODE])

def on_channel_view_change(ch_index):
    new_title = ALL_COLOR_SPACES[COLOR_MODE][ch_index]
    new_img = input_mod[:, :, ch_index]

    # Update the title
    channel_plot.setTitle(new_title)

    # Update the image
    channel_plot.set_image(new_img)
    # img_item = plotItem.items[0]
    # img_item.setImage(new_img)

    # Update the scatterplot
    new_scatter = make_pos_to_color_scatterplot(input_img, COLOR_MODE, ch_index)
    glvw_channel.plt_item.setData(pos=new_scatter.pos, color=new_scatter.color)

channel_cbox.currentIndexChanged.connect(on_channel_view_change)

# Setup color space combo box
color_space_cbox = QtGui.QComboBox()
color_space_cbox.addItems(ALL_COLOR_SPACES.keys())

def on_color_space_change(cspace_index):
    pass
    # COLOR_MODE = list(ALL_COLOR_SPACES.keys())[cspace_index]
    # input_mod = input_img[COLOR_MODE]
    # on_channel_view_change(cspace_index)


color_space_cbox.currentIndexChanged.connect(on_color_space_change)

# Setup widgets according to given grid layout
layoutgb = QtGui.QGridLayout()

win.setLayout(layoutgb)

layoutgb.addWidget(orig_img_plot, 1, 0)
layoutgb.addWidget(glvw_color_vis, 2, 0)

layoutgb.addWidget(channel_plot, 1, 1)
layoutgb.addWidget(glvw_channel, 2, 1)

layoutgb.addWidget(color_space_cbox, 0, 0)
layoutgb.addWidget(channel_cbox, 0, 1)


# Contrast/color control
# hist_lut_widget = pg.HistogramLUTWidget(image=pGray.plotItem.items[0])
# # hist_lut_widget.setImageItem()
# hist_lut_widget.autoHistogramRange()
# layoutgb.addWidget(hist_lut_widget, 0, 3, 3, 1)


# # build isocurves from smoothed data
# iso.setData(pg.gaussianFilter(data, (2, 2)))


win.resize(layoutgb.sizeHint())
main_area.setWidget(win)
main_area.resize(layoutgb.sizeHint() + QtCore.QSize(10, 10))
main_area.show()
# main_area.showMaximized()


# # Callbacks for handling user interaction
# def updatePlot():
#     global img, roi, data, p2
#     selected = roi.getArrayRegion(data, img)
#     p2.plot(selected.mean(axis=0), clear=True)

# roi.sigRegionChanged.connect(updatePlot)
# updatePlot()

# def updateIsocurve():
#     global isoLine, iso
#     iso.setLevel(isoLine.value())

# isoLine.sigDragged.connect(updateIsocurve)

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
