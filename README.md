# bayespecon

[![Continuous Integration](https://github.com/knaaptime/bayespecon/actions/workflows/unittests.yml/badge.svg)](https://github.com/knaaptime/bayespecon/actions/workflows/unittests.yml)
[![codecov](https://codecov.io/gh/knaaptime/bayespecon/branch/main/graph/badge.svg?token=XO4SilfBEb)](https://codecov.io/gh/knaaptime/bayespecon)


**Bayesian Spatial Econometric Models**

Because Bayesians are cool but MATLAB is not

Bayesian spatial models have excellent reference implementations in proprietary languages (e.g. MATLAB), but considerably less exposure in open-source languages. Python has a robust ecosystem for Bayesian inference, but fitting regression models with spatial effects traditionally requires a thorough knowledge of *both* probabilistic programming languages and spatial econometrics

Thus, `bayespecon` provides a simple interface for generating `pymc.Model` objects that fit commonly-used spatial econometric models. This design makes relatively sophisticated models available to applied researchers in short order, giving the benefit of fully Bayesian estimation without learning a new syntax.

Models are specified using the familiar Wilkinson format via [`formulaic`](https://matthew.wardrop.casa/formulaic/latest/) (but you can pass design matrices if you prefer), and spatial weights matrices $W$ are represented by PySAL [`graph`](https://pysal.org/libpysal/stable/generated/libpysal.graph.Graph.html#libpysal.graph.Graph) objects (or sparse matrices if you prefer). Estimation is handled by [`pymc`](https://www.pymc.io/welcome.html)

To ensure correctness, the vast majority of functionality is ported from and validated against Jim LeSage's [spatial econometrics toolbox](https://www.spatial-econometrics.com/).

**Main Features**:

- Fit models using Wilkinson formulas and PySAL `Graph` objects
- Classes for a wide variety of spatial econometric model specifications
- Models compile to PyMC for full customizability and [performant sampling options]()
- Fast [log-determinant functions]() for evaluating spatial terms
- Compute marginal (direct and indirect) effects for models with spatial terms
- Full suite of spatial diagnostics
- Functions to generate synthetic datasets using a known DGP for each model

## Supported Models

### Cross Sectional Models

#### SLX

$$y = X\beta + WX\theta + \epsilon$$

#### SAR

$$y = \rho Wy + X\beta + \epsilon$$

#### SEM

$$y = X\beta + u, u = \lambda Wu + \epsilon$$

#### SDM

$$y = \rho Wy + X\beta + WX\theta + \epsilon$$

#### SDEM

$$y = X\beta + WX\theta + u,  u = \lambda Wu + \epsilon$$

### Panel Models

#### OLS panel

$$y_{it} = x_{it}' \beta + a_i + \tau_t + \epsilon_{it}$$

#### SAR panel

$$y_{it} = \rho Wy_{it} + x_{it} ' \beta + a_i + \tau_t + \epsilon_{it}$$

#### SEM panel

$$y_{it} = x_{it}' \beta + a_i + \tau_t + u_{it}, u_{it}=\lambda Wu_{it} + \epsilon_{it}$$

#### SDM panel

$$y_{it} = \rho Wy_{it} + x_{it} ' \beta + Wx_{it}' \theta + a_i + \tau_t + \epsilon_{it}$$

#### SDEM panel

$$y_{it} = x_{it} ' \beta + Wx_{it}' \theta + a_i + \tau_t + u_{it}, u_{it}=\lambda Wu_{it} + \epsilon_{it}$$

#### OLS panel (Random Effects)

$$y_{it} = x_{it}' \beta + \alpha_i + \tau_t + \epsilon_{it}, \quad \alpha_i \sim N(0, \sigma_\alpha^2)$$

#### SAR panel (Random Effects)

$$y_{it} = \rho W y_{it} + x_{it}' \beta + \alpha_i + \tau_t + \epsilon_{it}, \quad \alpha_i \sim N(0, \sigma_\alpha^2)$$

#### SEM panel (Random Effects)

$$y_{it} = x_{it}' \beta + \alpha_i + \tau_t + u_{it}, \quad u_{it} = \lambda W u_{it} + \epsilon_{it}, \quad \alpha_i \sim N(0, \sigma_\alpha^2)$$


### Dynamic Panel Models

#### DLMPanelFE (Dynamic Linear Model, FE)

$$y_{it} = \phi y_{i, t-1} + x_{it}' \beta + a_i + \tau_t + \epsilon_{it}$$

#### SDMRPanelFE (Dynamic Restricted Spatial Durbin, FE)

$$y_{it} = \phi y_{i, t-1} + \rho W y_{it} - \rho \phi W y_{i, t-1} + x_{it}' \beta + W x_{it}' \theta + a_i + \tau_t + \epsilon_{it}$$

#### SDMUPanelFE (Dynamic Unrestricted Spatial Durbin, FE)

$$y_{it} = \phi y_{i, t-1} + \rho W y_{it} + \theta W y_{i, t-1} + x_{it}' \beta + W x_{it}' \theta + a_i + \tau_t + \epsilon_{it}$$


### Non-Linear Models

#### Spatial Probit

For binary outcomes $y_i \in \{0,1\}$, latent utilities $y^*$ follow a spatial lag:

$$y^* = \rho W y^* + X\beta + a + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$

where $a$ are region-level random effects and $y_i = \mathbf{1}[y^*_i > 0]$.

#### SAR Tobit

For censored outcomes, the spatial autoregressive Tobit specifies:

$$y^* = \rho W y^* + X\beta + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2 I)$$
$$y_i = \max(c,\, y^*_i)$$

where $c$ is the censoring threshold (default 0).

#### SEM Tobit

The spatial error Tobit places spatial dependence in the disturbance:

$$y^* = X\beta + u, \quad u = \lambda W u + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2 I)$$
$$y_i = \max(c,\, y^*_i)$$

### Non-Linear Panel Models

#### SARPanelTobit

spatial lag in the latent outcome:

$$y_t^* = \rho W y_t^* + X_t \beta + \varepsilon_t$$

#### SEMPanelTobit

spatial lag in the disturbance:

$$y^*_t = X_t \beta + u_t,  u_t = \lambda W u_t + \varepsilon_t$$
