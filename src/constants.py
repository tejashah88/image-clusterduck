ALL_COLOR_SPACES = [ 'RGB', 'YUV', 'YCrCb', 'LAB', 'LUV', 'HSV', 'XYZ' ]

COLOR_SPACE_LABELS = {
    'RGB': ('Red', 'Green', 'Blue'),
    'YUV': ('Luma', 'U', 'V'),
    'YCrCb': ('Luma', 'Cr', 'Cb'),
    'LAB': ('Lightness', 'A', 'B'),
    'LUV': ('L', 'U', 'V'),
    'HSV': ('Hue', 'Saturation', 'Value'),
    'XYZ': ('X', 'Y', 'Z'),
}

# Source: https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56
SUPPORTED_IMG_EXTS = {
    'Common Image files'        : ['jpg', 'jpeg', 'png'],
    'Windows bitmaps'           : ['bmp', 'dib'],
    'JPEG files'                : ['jpg', 'jpeg', 'jpe'],
    'JPEG 2000 files'           : ['jp2'],
    'Portable Network Graphics' : ['png'],
    'WebP'                      : ['webp'],
    'Portable image format'     : ['pbm', 'pgm', 'ppm', 'pxm', 'pnm'],
    'Sun rasters'               : ['sr', 'ras'],
    'TIFF files'                : ['tiff', 'tif'],
    'OpenEXR Image files'       : ['exr'],
    'Radiance HDR'              : ['hdr', 'pic'],
}

ALL_SUPPORTED_IMG_EXTS = tuple(set(sum([exts for exts in SUPPORTED_IMG_EXTS.values()], [])))