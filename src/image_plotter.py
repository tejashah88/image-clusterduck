import pyqtgraph as pg

HANDLE_SIZE = 10
ROI_PEN_WIDTH = 2

# A plot area (ViewBox + axes) for displaying the image
class ImagePlotter(pg.PlotWidget):
    def __init__(self, title='', size=(600, 450), img=None, enable_roi=False):
        # Create and initialize plotting widget
        super().__init__()

        # Always zoom view whenever image is changed
        self.enableAutoRange()

        # Create vars for the image and ROI items
        self.img_item = None
        self.roi_item = None

        # Create listener variable for containing the listeners for the image item
        self.click_listeners = []

        # Ensure that the pixels are square-looking
        self.setAspectLocked(True)

        # Setup the title and plot size
        self.setTitle(title)
        self.setFixedSize(*size)

        if img is not None:
            self.set_image(img)

        if enable_roi:
            self.enable_roi_rect()


    def set_image(self, img):
        self.img = img

        # Remove the current image item
        if self.img_item is not None:
            self.removeItem(self.img_item)

        # Create image item and add to plot
        self.img_item = pg.ImageItem(image=self.img)
        self.addItem(self.img_item)

        # Resize the plot so that it fits the whole image
        self.autoRange()

        # Flip image to match with image coordinates
        self.invertY()

        # On click on the image, emit the pixel location and value as an event
        orig_mouse_press_fn = self.img_item.mousePressEvent

        def get_pixel_at_pos(event):
            orig_mouse_press_fn(event)

            pixel_loc = event.pos().toPoint()
            x, y = pixel_loc.x(), pixel_loc.y()
            height, width = self.img.shape[:2]

            if ((x >= 0 and x < width) and (y >= 0 and y < height)):
                for listener in self.click_listeners:
                    listener(x, y, self.img[y, x])

        self.img_item.mousePressEvent = get_pixel_at_pos

        if self.roi_enabled:
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

            # Add image scaler handles
            self.roi_item.addScaleHandle([0.5, 1], [0.5, 0.5]).pen.setWidth(ROI_PEN_WIDTH)
            self.roi_item.addScaleHandle([1, 0.5], [0.5, 0.5]).pen.setWidth(ROI_PEN_WIDTH)

            self.addItem(self.roi_item)
            self.roi_item.setZValue(10)


    def disable_roi_rect(self):
        if self.roi_item is not None:
            self.removeItem(self.roi_item)
            self.roi_item = None


    def add_click_listener(self, click_fn):
        self.click_listeners.append(click_fn)


    def remove_click_listener(self, click_fn):
        self.click_listeners.remove(click_fn)


    def remove_all_click_listeners(self, click_fn):
        self.click_listeners = []
