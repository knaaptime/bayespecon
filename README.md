# bayespecon

[![Continuous Integration](https://github.com/knaaptime/bayespecon/actions/workflows/unittests.yml/badge.svg)](https://github.com/knaaptime/bayespecon/actions/workflows/unittests.yml)
[![codecov](https://codecov.io/gh/knaaptime/bayespecon/branch/main/graph/badge.svg?token=XO4SilfBEb)](https://codecov.io/gh/knaaptime/bayespecon)


**Bayesian Spatial Econometric Models**

The `bayespecon` package is designed to make it simpler to fit, diagnose, and interpret Bayesian spatial econometric regression models. It provides a suite of classes for building commmonly-used models using a straightforward API. Each model is implemented as a class that defines how spatial effects are represented, and the 'main' portion of the model specification is given using the familiar Wilkinson format via [`formulaic`](https://matthew.wardrop.casa/formulaic/latest/) (but you can pass design matrices if you prefer).

Each model class uses PySAL [`graph`](https://pysal.org/libpysal/stable/generated/libpysal.graph.Graph.html#libpysal.graph.Graph) objects to represent spatial weights, $W$, (or sparse matrices if you prefer) providing thorough integration with the scientific Python and spatial analysis ecosystems. Estimation is handled by [`pymc`](https://www.pymc.io/welcome.html).

This design makes it simple to build and iterate on spatial regression models using a straightforward notation while retaining all the benefits of a Bayesian framework. The resulting `pymc.Model` object is augmented to include the (correct) log-likelihood if requested, facilitating the use of Bayes Factors in model specification searches. The package also implements a suite of novel Bayesian spatial diagnostics.

Because models are compiled to PyMC, you can use the classes to specify a common model, then inspect the `pymc_model` object to sketch out a more complex specification. 
**Main Features**:

- Wide variety of spatial econometric models using Wilkinson formulas and PySAL `Graph` objects
- [Bayesian spatial diagnostics](https://www.sciencedirect.com/science/article/abs/pii/S0304407620303018)
- Marginal (direct and indirect) effects for models with spatial terms
- Fast [log-determinant functions]() for evaluating spatial terms
- Models compile to PyMC for full customizability
  - [performant sampling options]()
  - Full suite of Bayesian spatial diagnostics through  [`arviz`](https://python.arviz.org/en/stable/)
- Generate synthetic datasets using a known data-generating process for each model

