import math
import multiprocessing

import numpy as np
import pandas as pd
import sklearn.cluster
import cv2

from pyqtgraph.Qt import QtCore, QtGui

from constants import *


ALLOWED_GUI_COMPONENTS = ['spinbox', 'dropdown', 'slider', 'checkbox']
PARAM_CONFIG_COLUMNS = ['name', 'internal_name', 'title', 'gui_component', 'default_val', 'metadata']
NUM_CPUS = multiprocessing.cpu_count()

INT_MAX = 2**31 - 1
INT_MIN = -2**31

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
        self.param_config = pd.DataFrame(_param_config, columns=PARAM_CONFIG_COLUMNS)
        self.init_clustering_params()


    def setup_settings_layout(self):
        settings_layout = QtGui.QGridLayout()

        for index, param_row in self.param_config.iterrows():
            param_name = param_row['name']
            param_internal_name = param_row['internal_name']
            param_title = param_row['title']
            param_gui_component = param_row['gui_component']
            param_default_val = param_row['default_val']
            param_metadata = param_row['metadata']
            param_transform_fn = param_row['transform_fn']

            settings_layout.addWidget(QtGui.QLabel(f'{param_title}:'), index, 0)
            setting_widget = None

            if param_gui_component == 'spinbox':
                min_val, max_val, step_val = param_metadata

                setting_widget = QtGui.QSpinBox()
                setting_widget.setMinimum(min_val)
                setting_widget.setMaximum(max_val)
                setting_widget.setSingleStep(step_val)
                setting_widget.setValue(param_default_val)

                gen_on_spinbox_value_changed = lambda pname: (lambda val: self.set_clustering_params({ pname: val }))
                on_spinbox_value_changed = gen_on_spinbox_value_changed(param_name)

                setting_widget.valueChanged.connect(on_spinbox_value_changed)
            elif param_gui_component == 'dropdown':
                choices = param_metadata

                setting_widget = QtGui.QComboBox()
                setting_widget.addItems(choices)

                gen_on_dropdown_value_changed = lambda pname, choices: (lambda index: self.set_clustering_params({ pname: choices[index] }))
                on_dropdown_value_changed = gen_on_dropdown_value_changed(param_name, choices)

                setting_widget.currentIndexChanged.connect(on_dropdown_value_changed)
            elif param_gui_component == 'slider':
                min_val, max_val, step_val = param_metadata

                setting_widget = QtGui.QSlider(QtCore.Qt.Horizontal)
                setting_widget.setMinimum(min_val)
                setting_widget.setMaximum(max_val)
                setting_widget.setSingleStep(step_val)
                setting_widget.setValue(param_default_val)

                gen_on_slider_value_changed = lambda pname: (lambda val: self.set_clustering_params({ pname: val }))
                on_slider_value_changed = gen_on_slider_value_changed(param_name)

                setting_widget.valueChanged.connect(on_slider_value_changed)
            elif param_gui_component == 'checkbox':
                setting_widget = QtGui.QCheckBox()
                setting_widget.setChecked(param_default_val)

                gen_on_checkbox_value_changed = lambda pname: (lambda val: self.set_clustering_params({ pname: val }))
                on_checkbox_value_changed = gen_on_checkbox_value_changed(param_name)

                setting_widget.toggled.connect(on_checkbox_value_changed)

            settings_layout.addWidget(setting_widget, index, 1)

        return settings_layout


    def init_clustering_params(self, params={}):
        for index, param_row in self.param_config.iterrows():
            param_name = param_row['name']
            param_curr_val = params.get(param_name)
            param_default_val = param_row['default_val']
            param_transform_fn = param_row['transform_fn'] or (lambda args: args)
            self.params[param_name] = param_transform_fn(param_curr_val if param_curr_val is not None else param_default_val)


    def set_clustering_params(self, params={}):
        for param_name in params:
            param_curr_val = params.get(param_name)
            param_row = self.param_config[self.param_config['name'] == param_name]
            param_default_val = param_row['default_val'].tolist()[0]
            param_transform_fn = param_row['transform_fn'].tolist()[0] or (lambda args: args)
            self.params[param_name] = param_transform_fn(param_curr_val if param_curr_val is not None else param_default_val)


    def run_clustering(self, cv_img, color_mode):
        cluster_params = {}

        for index, param_row in self.param_config.iterrows():
            param_name = param_row['name']
            param_internal_name = param_row['internal_name']
            cluster_params[param_internal_name] = self.params[param_name]

        self.clusterer.set_params(**cluster_params)

        color_coords = cv_img[color_mode].reshape(-1, 3)
        rgb_image = cv_img.RGB.reshape(-1, 3)
        cluster_results = self.clusterer.fit(color_coords)
        return cluster_results


class KMeansImageClusterer(BaseImageClusterer):
    # Docs: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    _param_config = [
        ('num_clusters', 'n_clusters', 'Number of clusters', 'spinbox' , 8       , (1, INT_MAX, 1)          , None),
        ('init_centers', 'init'      , 'Initial centers'   , 'dropdown', 'random', ('random', 'kmeans++')   , None),
        ('repeat_count', 'n_init'    , 'Number of runs'    , 'spinbox' , 10      , (1, INT_MAX, 1)          , None),
        ('max_iter'    , 'max_iter'  , 'Max interations'   , 'spinbox' , 300     , (1, INT_MAX, 1)          , None),
        ('tolerance'   , 'tol'       , 'Tolerance'         , 'slider'  , -4      , (-10, 10, 1)             , lambda val: 10**val),
        ('num_jobs'    , 'n_jobs'    , 'Number of jobs'    , 'spinbox' , 1       , (1, NUM_CPUS, 1)         , None),
        ('algorithm'   , 'algorithm' , 'Algorithm type'    , 'dropdown', 'auto'  , ('auto', 'full', 'elkan'), None),
        ('verbose'     , 'verbose'   , 'Verbose Logging'   , 'checkbox', False   , None                     , None),
    ]

    def __init__(self):
        super().__init__(sklearn.cluster.KMeans, self._param_config)


    def run_clustering(self, cv_img, color_mode):
        cluster_results = super().run_clustering(cv_img, color_mode)

        color_centers = cluster_results.cluster_centers_
        color_labels = cluster_results.labels_
        rgb_colored_centers = get_rgb_from(color_centers, color_mode)
        cluster_error = cluster_results.inertia_
        num_iterations = cluster_results.n_iter_

        return (color_centers, color_labels, rgb_colored_centers, cluster_error, num_iterations)


class MiniBatchKMeansImageClusterer(BaseImageClusterer):
    # Docs: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html
    _param_config = [
        ('num_clusters'  , 'n_clusters'        , 'Number of clusters'   , 'spinbox' , 8       , (1, INT_MAX, 1)       , None),
        ('init_centers'  , 'init'              , 'Initial centers'      , 'dropdown', 'random', ('random', 'kmeans++'), None),
        ('repeat_count'  , 'n_init'            , 'Number of runs'       , 'spinbox' , 10      , (1, INT_MAX, 1)       , None),
        ('max_iter'      , 'max_iter'          , 'Max interations'      , 'spinbox' , 300     , (1, INT_MAX, 1)       , None),
        ('tolerance'     , 'tol'               , 'Tolerance'            , 'slider'  , -4      , (-10, 10, 1)          , lambda val: 10**val),
        ('batch_size'    , 'batch_size'        , 'Batch Size'           , 'spinbox' , 100     , (1, INT_MAX, 1)       , None),
        ('iter_plateau'  , 'max_no_improvement', 'Max iteration plateau', 'spinbox' , 10      , (0, INT_MAX, 1)       , None),
        ('reassign_ratio', 'reassignment_ratio', 'Reassignment Ratio'   , 'spinbox' , 10      , (0, 1000, 1)          , None),
        ('verbose'       , 'verbose'           , 'Verbose Logging'      , 'checkbox', False   , None                  , None),
    ]

    def __init__(self):
        super().__init__(sklearn.cluster.MiniBatchKMeans, self._param_config)


    def run_clustering(self, cv_img, color_mode):
        cluster_results = super().run_clustering(cv_img, color_mode)

        color_centers = cluster_results.cluster_centers_
        color_labels = cluster_results.labels_
        rgb_colored_centers = get_rgb_from(color_centers, color_mode)
        cluster_error = cluster_results.inertia_
        num_iterations = cluster_results.n_iter_

        return (color_centers, color_labels, rgb_colored_centers, cluster_error, num_iterations)


