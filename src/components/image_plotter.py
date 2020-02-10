import pyqtgraph as pg

HANDLE_SIZE = 10
ROI_PEN_WIDTH = 2


class ImagePlotter(pg.PlotWidget):
    """ A wrapper for pg.PlotWidget that allows easy ROI integration, image switching and pixel-click events. """
    def __init__(self, title='', size=(600, 450), img=None, enable_roi=False, enable_crosshair=False):
        # Create and initialize plotting widget
        super().__init__()

        # Create vars for the image, ROI, and crosshair items
        self.img_item = None
        self.roi_item = None
        self.vLine = None
        self.hLine = None

        # Do not automatically set the range of the image to prevent ROI from messing with it
        self.disableAutoRange()

        # Ensure that the pixels are square-looking
        self.setAspectLocked(True)

        # Setup the title and plot size
        self.setTitle(title)
        self.setFixedSize(*size)

        if img is not None:
            self.set_image(img)

        if enable_roi:
            self.enable_roi_rect()

        if enable_crosshair:
            self.enable_crosshairs()


    def set_image(self, img, auto_range=True):
        self.img = img

        if self.img_item is not None:
            # Replace the current image with the new one
            self.img_item.setImage(image=self.img)
        else:
            # Create image item and add to plot
            self.img_item = pg.ImageItem(image=self.img)
            self.addItem(self.img_item)

        # Resize the plot so that it fits the whole image
        if auto_range:
            self.autoRange()

        # Flip image to match with image coordinates
        self.invertY()

        if self.roi_enabled:
            # Set ROI dimensions to match image dimensions
            self.roi_item.setPos([0, 0])
            height, width = self.img.shape[:2]
            self.roi_item.setSize([width, height])


    @property
    def roi_enabled(self):
        return self.roi_item is not None


    def enable_roi_rect(self):
        if self.roi_item is None:
            # Add rectangular ROI object with draggable handles on all four sides
            height, width = self.img.shape[:2]
            pen = pg.mkPen(color=(255, 0, 200), width=ROI_PEN_WIDTH)
            self.roi_item = pg.ROI([0, 0], [width, height], pen=pen)
            self.roi_item.handleSize = HANDLE_SIZE

            # Add image scale handles on all 4 corners
            self.roi_item.addScaleHandle([1, 1], [0, 0]).pen.setWidth(ROI_PEN_WIDTH)
            self.roi_item.addScaleHandle([1, 0], [0, 1]).pen.setWidth(ROI_PEN_WIDTH)
            self.roi_item.addScaleHandle([0, 1], [1, 0]).pen.setWidth(ROI_PEN_WIDTH)
            self.roi_item.addScaleHandle([0, 0], [1, 1]).pen.setWidth(ROI_PEN_WIDTH)

            self.addItem(self.roi_item)
            self.roi_item.setZValue(10)


    def disable_roi_rect(self):
        if self.roi_item is not None:
            self.removeItem(self.roi_item)
            self.roi_item = None


    def enable_crosshairs(self):
        if self.vLine is None and self.hLine is None:
            self.vLine = pg.InfiniteLine(angle=90, movable=False)
            self.hLine = pg.InfiniteLine(angle=0, movable=False)

            self.addItem(self.vLine, ignoreBounds=True)
            self.addItem(self.hLine, ignoreBounds=True)

            def mouse_moved(mouse_pos):
                if self.sceneBoundingRect().contains(mouse_pos):
                    mouse_point = self.getViewBox().mapSceneToView(mouse_pos)
                    self.vLine.setPos(mouse_point.x())
                    self.hLine.setPos(mouse_point.y())

            self.scene().sigMouseMoved.connect(mouse_moved)

    def disable_crosshairs(self):
        if None not in [self.vLine, self.hLine]:
            self.scene().sigMouseMoved.disconnect()

            self.removeItem(self.vLine, ignoreBounds=True)
            self.removeItem(self.hLine, ignoreBounds=True)

            self.vLine = None
            self.hLine = None
