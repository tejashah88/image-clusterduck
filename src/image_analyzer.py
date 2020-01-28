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
def make_image_item(title, img_data):
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

    # Add rectangular ROI object with draggable handles on all four sides
    height, width = img_data.shape[:2]
    roi = pg.ROI([0, 0], [width, height])
    roi.addScaleHandle([0.5, 1], [0.5, 0.5])
    roi.addScaleHandle([1, 0.5], [0.5, 0.5])
    plt_widget.addItem(roi)
    roi.setZValue(10)

    plt_widget.setMouseEnabled(x=False, y=False)
    plt_widget.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)

    plt_widget.setTitle(title)
    plt_widget.setFixedSize(400, 300)

    # plt_widget.enableAutoRange()
    plt_widget.hideButtons()

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
    rgb_img = cv_img.RGB
    rows, cols, depth = np.shape(converted_img)
    r_arr, c_arr = np.mgrid[0:rows, 0:cols]

    for r, c in zip(r_arr.flatten(), c_arr.flatten()):
        color_conv = converted_img[r, c].tolist()
        color_rgb = rgb_img[r, c].tolist()

        color_pixel = [r, c, color_conv[ch_index]]

        pos += [color_pixel]
        color += [color_rgb]

    pos_arr = np.array(pos) / 255 * scale_factor
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

# Link all ROI objects within all given plots for consistent ROI analysis
def setup_roi_links(all_plots):
    all_rois = []
    for plot in all_plots:
        all_rois += [plot.plotItem.items[1]]

    def update_roi(src_roi):
        for roi in all_rois:
            if roi.pos() != src_roi.pos():
                roi.setPos(src_roi.pos())

    for roi in all_rois:
        roi.sigRegionChanged.connect(update_roi)

color_mode = 'HSV'

img = cv2.imread('./test-images/adorable-duck.jpg')

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

glvw_color_vis = make_3d_plot(make_img_scatterplot(input_img, color_mode))
glvw_ch1 = make_3d_plot(make_pos_to_color_scatterplot(input_img, color_mode, 0))
glvw_ch2 = make_3d_plot(make_pos_to_color_scatterplot(input_img, color_mode, 1))
glvw_ch3 = make_3d_plot(make_pos_to_color_scatterplot(input_img, color_mode, 2))

# # Custom ROI for selecting an image region
# roi = pg.ROI([-8, 14], [6, 5])
# roi.addScaleHandle([0.5, 1], [0.5, 0.5])
# roi.addScaleHandle([0, 0.5], [0.5, 0.5])
# p1.addItem(roi)
# roi.setZValue(10)  # make sure ROI is drawn above image

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

ch1 = make_image_item(channels[color_mode][0], input_mod[:, :, 0])
ch2 = make_image_item(channels[color_mode][1], input_mod[:, :, 1])
ch3 = make_image_item(channels[color_mode][2], input_mod[:, :, 2])
pAll = make_image_item('Original', input_rgb)
pGray = make_image_item('Gray', input_gray)

setup_axes_links(pAll, [pGray, ch1, ch2, ch3])
setup_roi_links([pAll, pGray, ch1, ch2, ch3])


# Setup widgets according to given grid layout
layoutgb = QtGui.QGridLayout()

win.setLayout(layoutgb)

layoutgb.addWidget(pAll, 0, 0)
# layoutgb.addWidget(pGray, 0, 1)
layoutgb.addWidget(glvw_color_vis, 1, 0)

layoutgb.addWidget(ch1, 0, 1)
layoutgb.addWidget(ch2, 0, 2)
layoutgb.addWidget(ch3, 0, 3)

layoutgb.addWidget(glvw_ch1, 1, 1)
layoutgb.addWidget(glvw_ch2, 1, 2)
layoutgb.addWidget(glvw_ch3, 1, 3)

# print(layoutgb.itemAtPosition(0, 2).expandingDirections())


# Contrast/color control
# hist_lut_widget = pg.HistogramLUTWidget(image=pGray.plotItem.items[0])
# # hist_lut_widget.setImageItem()
# hist_lut_widget.autoHistogramRange()
# layoutgb.addWidget(hist_lut_widget, 0, 3, 3, 1)



# # build isocurves from smoothed data
# iso.setData(pg.gaussianFilter(data, (2, 2)))

# screen_size = app.desktop().screenGeometry()
# finalWidth = screen_size.width()
# finalHeight = screen_size.height() + 400
# print(finalWidth, finalHeight)
# win.setFixedSize(finalWidth, finalHeight)
win.resize(layoutgb.sizeHint())
main_area.setWidget(win)
main_area.resize(layoutgb.sizeHint() + QtCore.QSize(10, 10))
main_area.show()
# main_area.showMaximized()



# def update():
#     ## update volume colors
#     global phase, scatter_plot, d2
#     s = -np.cos(d2*2+phase)
#     color = np.empty((len(d2),4), dtype=np.float32)
#     color[:,3] = np.clip(s * 0.1, 0, 1)
#     color[:,0] = np.clip(s * 3.0, 0, 1)
#     color[:,1] = np.clip(s * 1.0, 0, 1)
#     color[:,2] = np.clip(s ** 3, 0, 1)
#     scatter_plot.setData(color=color)
#     phase -= 0.1

# t = QtCore.QTimer()
# t.timeout.connect(update)
# t.start(0)


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
