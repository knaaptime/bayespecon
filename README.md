# bayespreg

Bayesian Spatial Econometric Regression Models

This package is a Python port of Jim LeSage's [spatial econometrics toolbox](https://www.spatial-econometrics.com/) with a few minor enhancements. Models are specified using the familiar Wilkinson format via [`formulaic`](https://matthew.wardrop.casa/formulaic/latest/) (but you can pass design matrixes if you prefer), and spatial weights matrices $W$ are represented by PySAL [`graph`](https://pysal.org/libpysal/stable/generated/libpysal.graph.Graph.html#libpysal.graph.Graph) objects (or sparse matrices if you prefer). Estimation is handled by [`pymc`](https://www.pymc.io/welcome.html)