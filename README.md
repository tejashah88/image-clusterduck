# image-clusterfuck
![Image analysis of Starry Night Sky](github-images/example-view.png "Image analysis of Starry Night Sky")

A Python GUI app for visualizing color distributions on images across various color spaces for image clustering.

## Features
* Supports 9 different color spaces to visualize
* All common image formats supported
* 3D color-based scatterplot for color distribution
* 3D channel-specific scatterplot
* Cropping and channel-specific thresholding
* Color cluster visualization using `scikit-learn` (WIP)
* Color histogram visualization (WIP)

## Installation
1. Clone this repo
2. Create conda environment from `.yml` file
3. Run `python src/image_analyzer.py`

## TODO
* Implement Clustering Visualization based on `scikit-learn`: https://scikit-learn.org/stable/modules/clustering.html
* Implement PCA analysis (probably) based on `scikit-learn`
* Implement Histogram Visualization