import math
import multiprocessing

import numpy as np
import pandas as pd
import sklearn.cluster
import cv2

# import h5py

from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.parametertree import Parameter, ParameterTree

from .constants import *


NUM_CPUS = multiprocessing.cpu_count()

INT_MAX = 2**31 - 1
INT_MIN = -2**31

# hdf5_store = h5py.File('./cache.hdf5', 'a')

def get_rgb_from(pixels, color_mode):
    if color_mode == 'RGB':
        return pixels

    pixels = pixels.reshape(1, len(pixels), 3).astype(np.uint8)

    if color_mode in ['HLS', 'HSV']:
        flag = getattr(cv2, f'COLOR_{color_mode}2RGB_FULL')
    else:
        flag = getattr(cv2, f'COLOR_{color_mode}2RGB')

    converted_pixels = cv2.cvtColor(pixels, flag)
    converted_pixels = np.squeeze(converted_pixels)
    return converted_pixels


class BaseImageClusterer:
    def __init__(self, clusterer_algo, _param_config):
        self.layout = None
        self.params = {}
        self.clusterer = clusterer_algo()
        self.param_config = _param_config
        self.set_clustering_params()


    def setup_settings_layout(self):
        settings_layout = QtGui.QGridLayout()

        pobj = Parameter.create(name='params', type='group', children=self.param_config)

        def on_param_change(param, changes):
            for param, change, data in changes:
                internal_name   = param.opts['iname']
                self.params[internal_name] = param.value()

        pobj.sigTreeStateChanged.connect(on_param_change)

        ptree = ParameterTree()
        ptree.setParameters(pobj, showTop=False)

        settings_layout.addWidget(ptree, 0, 0)
        return settings_layout


    def set_clustering_params(self, params={}):
        for param_row in self.param_config:
            param_name = param_row['iname']
            param_curr_val = params.get(param_name)
            param_default_val = param_row['value']
            self.params[param_name] = param_curr_val if param_curr_val is not None else param_default_val


    def run_clustering(self, cv_img, color_mode, crop_bounds=None):
        cluster_params = {}

        for param_row in self.param_config:
            param_name = param_row['iname']
            cluster_params[param_name] = self.params[param_name]

        self.clusterer.set_params(**cluster_params)

        initial_img = cv_img[color_mode]
        if crop_bounds is not None:
            x_min, y_min, x_max, y_max = crop_bounds
            initial_img = initial_img[y_min:y_max, x_min:x_max]

        color_coords = initial_img.reshape(-1, 3)
        rgb_image = cv_img.RGB.reshape(-1, 3)
        cluster_results = self.clusterer.fit(color_coords)
        return cluster_results


class KMeansImageClusterer(BaseImageClusterer):
    # Docs: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    _param_config = [
        {'name': 'Number of clusters', 'type': 'int' , 'value': 8       , 'limits': (1, INT_MAX)             , 'iname': 'n_clusters'},
        {'name': 'Initial centers'   , 'type': 'list', 'value': 'random', 'values': ['random', 'k-means++']  , 'iname': 'init'      },
        {'name': 'Number of runs'    , 'type': 'int' , 'value': 10      , 'limits': (1, INT_MAX)             , 'iname': 'n_init'    },
        {'name': 'Max iterations'    , 'type': 'int' , 'value': 300     , 'limits': (1, INT_MAX)             , 'iname': 'max_iter'  },
        {'name': 'Tolerance'         , 'type': 'int' , 'value': 1e-4    , 'limits': (1e-10, 1e+10)           , 'iname': 'tol'       , 'dec': True},
        {'name': 'Number of jobs'    , 'type': 'int' , 'value': 1       , 'limits': (1, NUM_CPUS)            , 'iname': 'n_jobs'    },
        {'name': 'Algorithm type'    , 'type': 'list', 'value': 'auto'  , 'values': ['auto', 'full', 'elkan'], 'iname': 'algorithm' },
        {'name': 'Verbose Logging'   , 'type': 'bool', 'value': False                                        , 'iname': 'verbose'   },
    ]

    def __init__(self):
        super().__init__(sklearn.cluster.KMeans, self._param_config)


    def run_clustering(self, cv_img, color_mode, crop_bounds=None):
        cluster_results = super().run_clustering(cv_img, color_mode, crop_bounds)

        color_centers = cluster_results.cluster_centers_
        color_labels = cluster_results.labels_
        rgb_colored_centers = get_rgb_from(color_centers, color_mode)
        cluster_error = cluster_results.inertia_
        num_iterations = cluster_results.n_iter_

        return (color_centers, color_labels, rgb_colored_centers, cluster_error, num_iterations)


class MiniBatchKMeansImageClusterer(BaseImageClusterer):
    # Docs: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html
    _param_config = [
        {'name': 'Number of clusters'   , 'type': 'int'  , 'value': 8       , 'limits': (1, INT_MAX)           , 'iname': 'n_clusters'        },
        {'name': 'Initial centers'      , 'type': 'list' , 'value': 'random', 'values': ['random', 'k-means++'], 'iname': 'init'              },
        {'name': 'Number of runs'       , 'type': 'int'  , 'value': 10      , 'limits': (1, INT_MAX)           , 'iname': 'n_init'            },
        {'name': 'Max iterations'       , 'type': 'int'  , 'value': 300     , 'limits': (1, INT_MAX)           , 'iname': 'max_iter'          },
        {'name': 'Tolerance'            , 'type': 'float', 'value': 1e-4    , 'limits': (1e-10, 1e+10)         , 'iname': 'tol'               , 'dec': True},
        ## FIXME: See above default value (0)
        {'name': 'Batch Size'           , 'type': 'int'  , 'value': 100     , 'limits': (1, INT_MAX)           , 'iname': 'batch_size'        },
        {'name': 'Max iteration plateau', 'type': 'int'  , 'value': 10      , 'limits': (0, INT_MAX)           , 'iname': 'max_no_improvement'},
        {'name': 'Reassignment Ratio'   , 'type': 'float', 'value': 0.01    , 'limits': (0, 1)                 , 'iname': 'reassignment_ratio', 'step': 0.001},
        {'name': 'Verbose Logging'      , 'type': 'bool' , 'value': False                                      , 'iname': 'verbose'           },
    ]

    def __init__(self):
        super().__init__(sklearn.cluster.MiniBatchKMeans, self._param_config)


    def run_clustering(self, cv_img, color_mode, crop_bounds=None):
        cluster_results = super().run_clustering(cv_img, color_mode, crop_bounds)

        color_centers = cluster_results.cluster_centers_
        color_labels = cluster_results.labels_
        rgb_colored_centers = get_rgb_from(color_centers, color_mode)
        cluster_error = cluster_results.inertia_
        num_iterations = cluster_results.n_iter_

        return (color_centers, color_labels, rgb_colored_centers, cluster_error, num_iterations)


class AffinityPropagationImageClusterer(BaseImageClusterer):
    # Docs: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html
    _param_config = [
        {'name': 'Damping Factor'       , 'type': 'float', 'value': 0.50    , 'limits': (0.50, 0.99), 'iname': 'damping'         , 'step': 0.001},
        {'name': 'Max iterations'       , 'type': 'int'  , 'value': 200     , 'limits': (1, INT_MAX), 'iname': 'max_iter'        },
        {'name': 'Max Convergence Iters', 'type': 'int'  , 'value': 15      , 'limits': (1, INT_MAX), 'iname': 'convergence_iter'},
        {'name': 'Verbose Logging'      , 'type': 'bool' , 'value': False                           , 'iname': 'verbose'         },
    ]

    def __init__(self):
        super().__init__(sklearn.cluster.AffinityPropagation, self._param_config)


    def run_clustering(self, cv_img, color_mode, crop_bounds=None):
        cluster_results = super().run_clustering(cv_img, color_mode, crop_bounds)

        color_centers = cluster_results.cluster_centers_
        color_labels = cluster_results.labels_
        rgb_color_centers = get_rgb_from(color_centers, color_mode)
        num_iterations = cluster_results.n_iter_

        return (color_centers, color_labels, rgb_color_centers, -1, num_iterations)


class MeanShiftImageClusterer(BaseImageClusterer):
    # Docs: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html
    _param_config = [
        {'name': 'Bandwidth'            , 'type': 'int' , 'value': -1   , 'limits': (-1, INT_MAX), 'iname': 'bandwidth'   },
        ## FIXME: See above default value (0)
        {'name': 'Binned Seeding'       , 'type': 'bool', 'value': False,                          'iname': 'bin_seeding' },
        {'name': 'Minimum Bin Frequency', 'type': 'int' , 'value': 1    , 'limits': (1, INT_MAX) , 'iname': 'min_bin_freq'},
        {'name': 'Cluster all points'   , 'type': 'bool', 'value': True ,                          'iname': 'cluster_all' },
        {'name': 'Number of jobs'       , 'type': 'int' , 'value': 1    , 'limits': (1, NUM_CPUS), 'iname': 'n_jobs'      },
        {'name': 'Max iterations'       , 'type': 'int' , 'value': 300  , 'limits': (1, INT_MAX) , 'iname': 'max_iter'    },
    ]

    def __init__(self):
        super().__init__(sklearn.cluster.MeanShift, self._param_config)


    def run_clustering(self, cv_img, color_mode, crop_bounds=None):
        cluster_results = super().run_clustering(cv_img, color_mode, crop_bounds)

        color_centers = cluster_results.cluster_centers_
        color_labels = cluster_results.labels_
        rgb_color_centers = get_rgb_from(color_centers, color_mode)
        num_iterations = cluster_results.n_iter_

        return (color_centers, color_labels, rgb_color_centers, -1, num_iterations)
