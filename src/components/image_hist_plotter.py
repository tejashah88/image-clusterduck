import cv2
import pyqtgraph as pg

HIST_COLORS = ('r', 'g', 'b')


class ImageHistPlotter(pg.PlotWidget):
    """ A wrapper for pg.PlotWidget that allows easy image histogram plotting. """
    def __init__(self, title='', size=(600, 450), img=None):
        # Create and initialize plotting widget
        super().__init__()

        # Setup the title and plot size
        self.setTitle(title)
        self.setFixedSize(*size)


    def plot_hist(self, img):
        # Clear and previous plots
        self.clear()

        # Calculate the histograms and plot them
        for i, color in enumerate(HIST_COLORS):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256]).squeeze()
            self.plot(hist, stepMode=not True, fillLevel=0, pen=color)

        # Resize the plot so that it fits within the view
        self.autoRange()
