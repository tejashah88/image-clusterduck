ALL_COLOR_SPACES = [ 'RGB', 'YUV', 'YCrCb', 'LAB', 'LUV', 'HLS', 'HSV', 'XYZ' ]

IMPLEMENTED_COLOR_SPACES = ['RGB', 'BGR', 'YUV', 'YCrCb', 'LAB', 'LUV', 'HLS', 'HSV', 'XYZ', 'GRAY']

COLOR_SPACE_LABELS = {
    'RGB': ('Red', 'Green', 'Blue'),
    'YUV': ('Luma', 'U', 'V'),
    'YCrCb': ('Luma', 'Cr', 'Cb'),
    'LAB': ('Lightness', 'A', 'B'),
    'LUV': ('L', 'U', 'V'),
    'HLS': ('Hue', 'Lightness', 'Saturation'),
    'HSV': ('Hue', 'Saturation', 'Value'),
    'XYZ': ('X', 'Y', 'Z'),
}

ALL_CLUSTER_ALGORITHMS = [
    'K-Means', 'Mini Batch K-Means', 'Affinity Propagation', 'Mean Shift',
    'Spectral Clustering', 'Ward Hierarchical Clustering',
    'Agglomerative Clustering', 'DBSCAN', 'OPTICS',
    'Gaussian Mixtures', 'Birch'
]