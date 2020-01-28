import numpy as np
import cv2

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from cv_img import CvImg

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

# Interpret image data as row-major instead of col-major
pg.setConfigOptions(imageAxisOrder='row-major')

app = pg.mkQApp()

main_area = pg.QtGui.QScrollArea()
main_area.setWindowTitle('Image Cluster Analysis')

win = pg.GraphicsLayoutWidget()

# A plot area (ViewBox + axes) for displaying the image
def make_image_item(title, img_data, enable_roi=False):
    # Create plot widget
    plt_widget = pg.PlotWidget()

    # Ensure that the pixels are square-looking
    plt_widget.setAspectLocked(True)

    # Create image item and add to plot
    img_item = pg.ImageItem(image=img_data)
    plt_widget.addItem(img_item)

    # Zoom plot to fit image
    plt_widget.autoRange()
    plt_widget.invertY()

    if enable_roi:
        # Add rectangular ROI object with draggable handles on all four sides
        height, width = img_data.shape[:2]
        pen = pg.mkPen(color=(255, 0, 200), width=2)
        roi = pg.ROI([0, 0], [width, height], pen=pen)
        roi.handleSize = 10

        roi.addScaleHandle([0.5, 1], [0.5, 0.5]).pen.setWidth(2)
        roi.addScaleHandle([1, 0.5], [0.5, 0.5]).pen.setWidth(2)

        plt_widget.addItem(roi)
        roi.setZValue(10)

    plt_widget.setTitle(title)
    plt_widget.setFixedSize(400, 300)

    plt_widget.enableAutoRange()
    # plt_widget.hideButtons()

    orig_mouse_press_fn = img_item.mousePressEvent
    def getPos(event):
        orig_mouse_press_fn(event)

        pixel_loc = event.pos().toPoint()
        x, y = pixel_loc.x(), pixel_loc.y()
        try:
            print(x, y, img_data[y, x])
        except:
            print('could not access img_data pixel info')

    img_item.mousePressEvent = getPos

    return plt_widget


def make_img_scatterplot(cv_img, color_mode, scale_factor=3):
    pos_arr = cv_img[color_mode].reshape(-1, 3) / 255 * scale_factor
    color_arr = cv_img.RGB.reshape(-1, 3) / 255

    splot = gl.GLScatterPlotItem(
        pos=pos_arr, color=color_arr,
        size=1, pxMode=True
    )

    splot.setGLOptions('opaque')
    return splot


def make_pos_to_color_scatterplot(cv_img, color_mode, ch_index, scale_factor=3):
    pos = []
    color = []

    converted_img = cv_img[color_mode]
    rows, cols = converted_img.shape[:2]
    rgb_img = cv_img.RGB
    rows, cols, depth = np.shape(converted_img)
    r_arr, c_arr = np.mgrid[0:rows, 0:cols]

    for r, c in zip(r_arr.flatten(), c_arr.flatten()):
        color_conv = converted_img[r, c].tolist()
        color_rgb = rgb_img[r, c].tolist()

        color_pixel = [r, c, color_conv[ch_index]]

        pos += [color_pixel]
        color += [color_rgb]

    pos_arr = np.array(pos)
    pos_arr = pos_arr / np.array([ max(rows, cols), max(rows, cols), max(pos_arr[:, 2]) ])
    pos_arr = pos_arr * np.array([ scale_factor, scale_factor, scale_factor / 2 ])
    color_arr = np.array(color) / 255

    splot = gl.GLScatterPlotItem(
        pos=pos_arr, color=color_arr,
        size=1, pxMode=True
    )

    splot.setGLOptions('opaque')

    return splot


def make_pixelbox(num_pixels=1000000, scale_factor=3):
    pos = np.random.random(size=(num_pixels, 3)) * (2 * scale_factor) - scale_factor
    color = np.random.random(size=(num_pixels, 4))
    size = np.random.random(size=num_pixels)

    pixel_box = gl.GLScatterPlotItem(pos=pos, color=color, size=size, pxMode=True)
    pixel_box.setGLOptions('opaque')

    return pixel_box


def make_3d_plot(plot_3d, enable_axes=True, scale_factor=3):
    glvw = gl.GLViewWidget()
    # glvw.setBackgroundColor(255, 255, 255)

    if enable_axes:
        # x = blue, y = yellow, z = green
        coord_axes = gl.GLAxisItem(size=QtGui.QVector3D(scale_factor, scale_factor, scale_factor))
        glvw.addItem(coord_axes)

    glvw.addItem(plot_3d)
    glvw.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)

    glvw.setFixedSize(400, 300)

    return glvw


# Link the image plot axes together for consistent panning and zooming
def setup_axes_links(leader_plot, follower_plots):
    for plot in follower_plots:
        plot.setXLink(leader_plot)
        plot.setYLink(leader_plot)


color_mode = 'RGB'

img = cv2.imread('./test-images/starry-night.jpg')

num_pixels = img.size
target_num_pixels = 1000000

resize_factor = 1 / ( (num_pixels / target_num_pixels) ** 0.5 )
resize_factor = min(resize_factor, 1)
if resize_factor < 1:
    print('Resize factor:', resize_factor)
    img = cv2.resize(img, None, fx=resize_factor, fy=resize_factor)
# img_data = cv2.resize(img_data, None, fx=.25, fy=.25)

input_img   = CvImg.from_ndarray(img)
input_rgb   = input_img.RGB
input_gray  = input_img.GRAY
input_mod   = input_img[color_mode]


print('Num pixels:', input_img.RGB.size)

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


channels = {
    'RGB': ('Red', 'Green', 'Blue'),
    'YUV': ('Luma (Brightness)', 'U (blue projection)', 'V (red projection)'),
    'YCrCb': ('Luma (Brightness)', 'Cr (Red from Luma)', 'Cb (Blue from Luma)'),
    'LAB': ('Lightness', 'A (Green to Red)', 'B (Blue to Yellow)'),
    'LUV': ('L (Brightness)', 'U', 'V'),
    'HLS': ('Hue', 'Lightness', 'Saturation'),
    'HSV': ('Hue', 'Saturation', 'Value'),
    'XYZ': ('X', 'Y', 'Z'),
}

DEFAULT_INDEX = 0

orig_img_plot = make_image_item('Original', input_rgb, enable_roi=True)
channel_plot = make_image_item(channels[color_mode][DEFAULT_INDEX], input_mod[:, :, DEFAULT_INDEX])

glvw_color_vis = make_3d_plot(make_img_scatterplot(input_img, color_mode))
glvw_channel = make_3d_plot(make_pos_to_color_scatterplot(input_img, color_mode, DEFAULT_INDEX))

# pGray = make_image_item('Gray', input_gray)

setup_axes_links(orig_img_plot, [channel_plot])

# Setup widgets according to given grid layout
layoutgb = QtGui.QGridLayout()

win.setLayout(layoutgb)

layoutgb.addWidget(orig_img_plot, 0, 0)
# layoutgb.addWidget(pGray, 0, 1)
layoutgb.addWidget(glvw_color_vis, 1, 0)

layoutgb.addWidget(channel_plot, 0, 1)
layoutgb.addWidget(glvw_channel, 1, 1)

channel_cbox = QtGui.QComboBox()
channel_cbox.addItems(channels[color_mode])

def on_channel_view_change(ch_index):
    new_title = channels[color_mode][ch_index]
    new_img = input_mod[:, :, ch_index]

    # Update the title
    channel_plot.setTitle(new_title)

    # Update the image
    img_item = channel_plot.plotItem.items[0]
    img_item.setImage(new_img)

    # Update the scatterplot
    new_scatter = make_pos_to_color_scatterplot(input_img, color_mode, ch_index)
    glvw_channel.items[1].setData(
        pos=new_scatter.pos, color=new_scatter.color,
        size=new_scatter.size, pxMode=new_scatter.pxMode
    )


channel_cbox.currentIndexChanged.connect(on_channel_view_change)


layoutgb.addWidget(channel_cbox, 2, 0)


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
