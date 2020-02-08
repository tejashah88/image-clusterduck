# image-clusterfuck
![Image analysis of Starry Night Sky](github-images/example-view.png "Image analysis of Starry Night Sky")

A Python GUI app for visualizing color distributions across various color spaces for image clustering. Supports cropping and thresholding.

## Features
* Supports 9 different color spaces to visualize
  * BGR, RGB, YUV, YCrCb, LAB, LUV, HLS, HSV, XYZ
* 3D scatterplots for color distribution and for selected channel
* Cropping and channel-specific thresholding
* Color cluster visualization using `scikit-learn` (WIP)
* Color histogram visualization (WIP)

## TODO
* Implement Clustering Visualization based on `scikit-learn`: https://scikit-learn.org/stable/modules/clustering.html
* Implement PCA analysis (probably) based on `scikit-learn`
* Implement Histogram Visualization