import cv2

IMPLEMENTED_COLOR_SPACES = ['RGB', 'BGR', 'YUV', 'YCrCb', 'LAB', 'LUV', 'HSV', 'XYZ', 'GRAY']

class CvImg:
    ''' A class for allowing easy access to image color conversion methods. '''

    def __init__(self):
        self.image = None


    def __getitem__(self, sname):
        ''' Utility function to allow access to different color-converted images via dictionary-like access. '''
        for space_name in IMPLEMENTED_COLOR_SPACES:
            if sname == space_name:
                return getattr(self, sname)
        return None


    @staticmethod
    def from_filename(file_path):
        ''' Static version of 'load_from_file'. '''
        return CvImg().load_from_file(file_path)


    def load_from_file(self, file_path):
        self.image = cv2.imread(file_path)
        ''' Allows loading an image from a file path. '''
        return self


    @staticmethod
    def from_ndarray(ndarray):
        ''' Static version of 'from_ndarray'. '''
        return CvImg().load_from_ndarray(ndarray)


    def load_from_ndarray(self, ndarray):
        self.image = ndarray.copy()
        ''' Allows loading an image from a numpy array. '''
        return self


    @property
    def BGR(self):
        ''' Get BGR format of image (native format). '''
        return self.image


    @property
    def RGB(self):
        ''' Get RGB format of image. '''
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)


    @property
    def YUV(self):
        ''' Get YUV format of image. '''
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2YUV)


    @property
    def YCrCb(self):
        ''' Get YCrCb format of image. '''
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2YCrCb)


    @property
    def LAB(self):
        ''' Get LAB format of image. '''
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)


    @property
    def LUV(self):
        ''' Get LUV format of image. '''
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2LUV)


    @property
    def HSV(self):
        ''' Get HSV format of image. '''
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV_FULL)


    @property
    def XYZ(self):
        ''' Get XYZ format of image. '''
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2XYZ)


    @property
    def GRAY(self):
        ''' Get grayscale format of image. '''
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
