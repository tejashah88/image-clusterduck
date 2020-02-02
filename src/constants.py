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