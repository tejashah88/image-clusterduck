import cv2

IMPLEMENTED_COLOR_SPACES = ['RGB', 'BGR', 'YUV', 'YCrCb', 'LAB', 'LUV', 'HLS', 'HSV', 'XYZ', 'GRAY']

class CvImg:
    def __init__(self):
        self.image = {}

    def __getitem__(self, sname):
        for space_name in IMPLEMENTED_COLOR_SPACES:
            if sname == space_name:
                return getattr(self, sname)
        return None

    @staticmethod
    def from_filename(filename):
        return CvImg().load_from_file(filename)

    def load_from_file(self, filename):
        self.image = cv2.imread(filename)
        return self

    @staticmethod
    def from_ndarray(ndarray):
        return CvImg().load_from_ndarray(ndarray)

    def load_from_ndarray(self, ndarray):
        self.image = ndarray.copy()
        return self

    @property
    def BGR(self):
        return self.image

    @property
    def RGB(self):
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

    @property
    def YUV(self):
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2YUV)

    @property
    def YCrCb(self):
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2YCrCb)

    @property
    def LAB(self):
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)

    @property
    def LUV(self):
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2LUV)

    @property
    def HLS(self):
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2HLS_FULL)

    @property
    def HSV(self):
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV_FULL)

    @property
    def XYZ(self):
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2XYZ)

    @property
    def GRAY(self):
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
