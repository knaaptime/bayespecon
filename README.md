# bayespreg

Bayesian Spatial Econometric Regression Models

This package is a Python port of Jim LeSage's [spatial econometrics toolbox](https://www.spatial-econometrics.com/) with a few minor enhancements. Models are specified using the familiar Wilkinson format via [`formulaic`](https://matthew.wardrop.casa/formulaic/latest/) (but you can pass design matrixes if you prefer), and spatial weights matrices $W$ are represented by PySAL [`graph`](https://pysal.org/libpysal/stable/generated/libpysal.graph.Graph.html#libpysal.graph.Graph) objects (or sparse matrices if you prefer). Estimation is handled by [`pymc`](https://www.pymc.io/welcome.html)

## Supported Models

### Cross Sectional Models

- **SLX**: $y = X\beta + WX\theta + \epsilon$
- **SAR**: $y = \rho Wy + X\beta + \epsilon$
- **SEM**: $y = X\beta + u$, $u = \lambda Wu + \epsilon$
- **SDM**: $y = \rho Wy + X\beta + WX\theta + \epsilon$
- **SDEM**: $y = X\beta + WX\theta + u$, $u = \lambda Wu + \epsilon$


### Panel Models

- OLS panel: $y_{it} = x_{it}'\beta + a_i + \tau_t + \epsilon_{it}$
- SAR panel: $y_{it} = \rho (Wy)_{it} + x_{it}'\beta + a_i + \tau_t + \epsilon_{it}$
- SEM panel: $y_{it} = x_{it}'\beta + a_i + \tau_t + u_{it},\; u_{it}=\lambda (Wu)_{it}+\epsilon_{it}$
- SDM panel: $y_{it} = \rho (Wy)_{it} + x_{it}'\beta + (Wx)_{it}'\theta + a_i + \tau_t + \epsilon_{it}$
- SDEM panel: $y_{it} = x_{it}'\beta + (Wx)_{it}'\theta + a_i + \tau_t + u_{it},\; u_{it}=\lambda (Wu)_{it}+\epsilon_{it}$

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

- **SARPanelTobit** — spatial lag in the latent outcome: $y^*_t = \rho W y^*_t + X_t \beta + \varepsilon_t$
- **SEMPanelTobit** — spatial lag in the disturbance: $y^*_t = X_t \beta + u_t$, $u_t = \lambda W u_t + \varepsilon_t$
