## Supported Models

### Cross Sectional Models

#### OLS

$$y = X\beta + \epsilon$$

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
#### SLX panel

$$y_{it} = x_{it}'\beta + Wx_{it}'\theta + a_i + \tau_t + \epsilon_{it}$$

#### OLS panel (Random Effects)

$$y_{it} = x_{it}' \beta + \alpha_i + \tau_t + \epsilon_{it}, \quad \alpha_i \sim N(0, \sigma_\alpha^2)$$

#### SAR panel (Random Effects)

$$y_{it} = \rho W y_{it} + x_{it}' \beta + \alpha_i + \tau_t + \epsilon_{it}, \quad \alpha_i \sim N(0, \sigma_\alpha^2)$$

#### SEM panel (Random Effects)

$$y_{it} = x_{it}' \beta + \alpha_i + \tau_t + u_{it}, \quad u_{it} = \lambda W u_{it} + \epsilon_{it}, \quad \alpha_i \sim N(0, \sigma_\alpha^2)$$


### Dynamic Panel Models

#### OLSPanelDynamic (Dynamic Linear Model)

$$y_{it} = \phi y_{i, t-1} + x_{it}' \beta + a_i + \tau_t + \epsilon_{it}$$

#### SDMRPanelDynamic (Dynamic Restricted Spatial Durbin)

$$y_{it} = \phi y_{i, t-1} + \rho W y_{it} - \rho \phi W y_{i, t-1} + x_{it}' \beta + W x_{it}' \theta + a_i + \tau_t + \epsilon_{it}$$

#### SDMUPanelDynamic (Dynamic Unrestricted Spatial Durbin)

$$y_{it} = \phi y_{i, t-1} + \rho W y_{it} + \theta W y_{i, t-1} + x_{it}' \beta + W x_{it}' \theta + a_i + \tau_t + \epsilon_{it}$$

#### SARPanelDynamic (Dynamic SAR)

$$y_{it} = \phi y_{i, t-1} + \rho W y_{it} + x_{it}' \beta + a_i + \tau_t + \epsilon_{it}$$

#### SEMPanelDynamic (Dynamic SEM)

$$y_{it} = \phi y_{i, t-1} + x_{it}' \beta + a_i + \tau_t + u_{it}, \quad u_{it} = \lambda W u_{it} + \epsilon_{it}$$

#### SDEMPanelDynamic (Dynamic SDEM)

$$y_{it} = \phi y_{i, t-1} + x_{it}' \beta + W x_{it}' \theta + a_i + \tau_t + u_{it}, \quad u_{it} = \lambda W u_{it} + \epsilon_{it}$$

#### SLXPanelDynamic (Dynamic SLX)

$$y_{it} = \phi y_{i, t-1} + x_{it}' \beta + W x_{it}' \theta + a_i + \tau_t + \epsilon_{it}$$


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

#### SDM Tobit

The spatial Durbin Tobit combines a spatial lag with spatially lagged covariates in the latent outcome:

$$y^* = \rho W y^* + X\beta + WX\theta + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2 I)$$
$$y_i = \max(c,\, y^*_i)$$

### Non-Linear Panel Models

#### SARPanelTobit

spatial lag in the latent outcome:

$$y_t^* = \rho W y_t^* + X_t \beta + \varepsilon_t$$

#### SEMPanelTobit

spatial lag in the disturbance:

$$y^*_t = X_t \beta + u_t,  u_t = \lambda W u_t + \varepsilon_t$$

### Flow Models

Origin-destination flow matrices $Y$ (size $n \times n$) are vectorized to $y$ of length $N = n^2$. The destination, origin, and origin-destination weight matrices are $W_d = I_n \otimes W$, $W_o = W \otimes I_n$, $W_w = W \otimes W$.

#### SARFlow

$$y = \rho_d W_d y + \rho_o W_o y + \rho_w W_w y + X\beta + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2 I)$$

#### SARFlowSeparable

Imposes the separability constraint $\rho_w = -\rho_d \rho_o$, enabling an exact $\mathcal{O}(n)$ log-determinant via eigenvalue factorization:

$$y = \rho_d W_d y + \rho_o W_o y - \rho_d\rho_o W_w y + X\beta + \varepsilon$$

#### PoissonSARFlow

For count-valued flows, the mean is filtered through the spatial multiplier $A(\boldsymbol{\rho}) = I - \rho_d W_d - \rho_o W_o - \rho_w W_w$:

$$y_{ij} \sim \text{Poisson}(\lambda_{ij}), \quad \log \boldsymbol{\lambda} = A(\boldsymbol{\rho})^{-1} X\beta$$

#### PoissonSARFlowSeparable

Poisson flow with separability constraint $\rho_w = -\rho_d \rho_o$:

$$y_{ij} \sim \text{Poisson}(\lambda_{ij}), \quad \log \boldsymbol{\lambda} = A(\boldsymbol{\rho})^{-1} X\beta, \quad \rho_w = -\rho_d\rho_o$$

### Panel Flow Models

Time-stacked versions of the flow models above. Observations are stacked time-first across $T$ periods. Poisson variants are restricted to pooled mode.

#### SARFlowPanel

$$y_t = \rho_d W_d y_t + \rho_o W_o y_t + \rho_w W_w y_t + X_t\beta + \varepsilon_t$$

#### SARFlowSeparablePanel

$$y_t = \rho_d W_d y_t + \rho_o W_o y_t - \rho_d\rho_o W_w y_t + X_t\beta + \varepsilon_t$$

#### PoissonSARFlowPanel

$$y_{ijt} \sim \text{Poisson}(\lambda_{ijt}), \quad \log \boldsymbol{\lambda}_t = A(\boldsymbol{\rho})^{-1} X_t\beta$$

#### PoissonSARFlowSeparablePanel

$$y_{ijt} \sim \text{Poisson}(\lambda_{ijt}), \quad \log \boldsymbol{\lambda}_t = A(\boldsymbol{\rho})^{-1} X_t\beta, \quad \rho_w = -\rho_d\rho_o$$
