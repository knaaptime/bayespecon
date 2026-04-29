# Supported Models

## Cross Sectional Models

### OLS

$$y = X\beta + \epsilon$$

### SLX

$$y = X\beta + WX\theta + \epsilon$$

### SAR

$$y = \rho Wy + X\beta + \epsilon$$

### SEM

$$y = X\beta + u, \quad u = \lambda Wu + \epsilon$$

### SDM

$$y = \rho Wy + X\beta + WX\theta + \epsilon$$

### SDEM

$$y = X\beta + WX\theta + u, \quad u = \lambda Wu + \epsilon$$

## Panel Models

### OLS panel

$$y_{it} = x_{it}' \beta + a_i + \tau_t + \epsilon_{it}$$

### SAR panel

$$y_{it} = \rho Wy_{it} + x_{it}' \beta + a_i + \tau_t + \epsilon_{it}$$

### SEM panel

$$y_{it} = x_{it}' \beta + a_i + \tau_t + u_{it}, \quad u_{it} = \lambda Wu_{it} + \epsilon_{it}$$

### SDM panel

$$y_{it} = \rho Wy_{it} + x_{it}' \beta + Wx_{it}' \theta + a_i + \tau_t + \epsilon_{it}$$

### SDEM panel

$$y_{it} = x_{it}' \beta + Wx_{it}' \theta + a_i + \tau_t + u_{it}, \quad u_{it} = \lambda Wu_{it} + \epsilon_{it}$$

### SLX panel

$$y_{it} = x_{it}' \beta + Wx_{it}' \theta + a_i + \tau_t + \epsilon_{it}$$

### OLS panel (Random Effects)

$$y_{it} = x_{it}' \beta + \alpha_i + \tau_t + \epsilon_{it}, \quad \alpha_i \sim N(0, \sigma_\alpha^2)$$

### SAR panel (Random Effects)

$$y_{it} = \rho W y_{it} + x_{it}' \beta + \alpha_i + \tau_t + \epsilon_{it}, \quad \alpha_i \sim N(0, \sigma_\alpha^2)$$

### SEM panel (Random Effects)

$$y_{it} = x_{it}' \beta + \alpha_i + \tau_t + u_{it}, \quad u_{it} = \lambda W u_{it} + \epsilon_{it}, \quad \alpha_i \sim N(0, \sigma_\alpha^2)$$

### SDEM panel (Random Effects)

$$y_{it} = x_{it}' \beta + W x_{it}' \theta + \alpha_i + u_{it}, \quad u_{it} = \lambda W u_{it} + \epsilon_{it}, \quad \alpha_i \sim N(0, \sigma_\alpha^2)$$

## Dynamic Panel Models

### OLSPanelDynamic (Dynamic Linear Model)

$$y_{it} = \phi y_{i,t-1} + x_{it}' \beta + a_i + \tau_t + \epsilon_{it}$$

### SDMRPanelDynamic (Dynamic Restricted Spatial Durbin)

$$y_{it} = \phi y_{i,t-1} + \rho W y_{it} - \rho \phi W y_{i,t-1} + x_{it}' \beta + W x_{it}' \theta + a_i + \tau_t + \epsilon_{it}$$

### SDMUPanelDynamic (Dynamic Unrestricted Spatial Durbin)

$$y_{it} = \phi y_{i,t-1} + \rho W y_{it} + \theta W y_{i,t-1} + x_{it}' \beta + W x_{it}' \theta + a_i + \tau_t + \epsilon_{it}$$

### SARPanelDynamic (Dynamic SAR)

$$y_{it} = \phi y_{i,t-1} + \rho W y_{it} + x_{it}' \beta + a_i + \tau_t + \epsilon_{it}$$

### SEMPanelDynamic (Dynamic SEM)

$$y_{it} = \phi y_{i,t-1} + x_{it}' \beta + a_i + \tau_t + u_{it}, \quad u_{it} = \lambda W u_{it} + \epsilon_{it}$$

### SDEMPanelDynamic (Dynamic SDEM)

$$y_{it} = \phi y_{i,t-1} + x_{it}' \beta + W x_{it}' \theta + a_i + \tau_t + u_{it}, \quad u_{it} = \lambda W u_{it} + \epsilon_{it}$$

### SLXPanelDynamic (Dynamic SLX)

$$y_{it} = \phi y_{i,t-1} + x_{it}' \beta + W x_{it}' \theta + a_i + \tau_t + \epsilon_{it}$$

## Non-Linear Models

### Spatial Probit

$$y^* = \rho W y^* + X\beta + a + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I), \quad y_i = \mathbf{1}[y_i^* > 0]$$

### Tobit (SAR Tobit)

$$y_i = \max(c, y_i^*), \quad y^* = \rho W y^* + X\beta + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2 I)$$

### Tobit (SEM Tobit)

$$y_i = \max(c, y_i^*), \quad y^* = X\beta + u, \quad u = \lambda Wu + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2 I)$$

### Tobit (SDM Tobit)

$$y_i = \max(c, y_i^*), \quad y^* = \rho W y^* + X\beta + WX\theta + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2 I)$$

### Panel Tobit (SAR)

$$y_{it} = \max(c, y_{it}^*), \quad y_t^* = \rho W y_t^* + X_t\beta + \varepsilon_t$$

### Panel Tobit (SEM)

$$y_{it} = \max(c, y_{it}^*), \quad y_t^* = X_t\beta + u_t, \quad u_t = \lambda W u_t + \varepsilon_t$$

## Flow Models

Vectorize the origin-destination flow matrix to $y \in \mathbb{R}^{N}$ with $N = n^2$, and define destination, origin, and network weight matrices as $W_d$, $W_o$, and $W_w$.

### OLSFlow

$$y = X\beta + \varepsilon$$

### PoissonFlow

$$y_{ij} \sim \operatorname{Poisson}(\lambda_{ij}), \quad \log \boldsymbol{\lambda} = X\beta$$

### SARFlow

$$y = \rho_d W_d y + \rho_o W_o y + \rho_w W_w y + X\beta + \varepsilon$$

### SARFlowSeparable

$$y = \rho_d W_d y + \rho_o W_o y - \rho_d \rho_o W_w y + X\beta + \varepsilon$$

### PoissonSARFlow

$$y_{ij} \sim \operatorname{Poisson}(\lambda_{ij}), \quad \log \boldsymbol{\lambda} = A(\boldsymbol{\rho})^{-1} X\beta$$

### PoissonSARFlowSeparable

$$y_{ij} \sim \operatorname{Poisson}(\lambda_{ij}), \quad \log \boldsymbol{\lambda} = A(\boldsymbol{\rho})^{-1} X\beta, \quad \rho_w = -\rho_d \rho_o$$

### SEMFlow

$$y = X\beta + u, \quad u = \lambda_d W_d u + \lambda_o W_o u + \lambda_w W_w u + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2 I)$$

### SEMFlowSeparable

$$y = X\beta + u, \quad u = \lambda_d W_d u + \lambda_o W_o u - \lambda_d \lambda_o W_w u + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2 I)$$

## Panel Flow Models

Stack the flow models above across $T$ periods in time-first order. The Poisson panel variants currently operate in pooled mode.

### OLSFlowPanel

$$y_t = X_t\beta + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma^2 I_N)$$

### PoissonFlowPanel

$$y_{ij,t} \sim \operatorname{Poisson}(\lambda_{ij,t}), \quad \log \boldsymbol{\lambda}_t = X_t\beta$$

### SARFlowPanel

$$y_t = \rho_d W_d y_t + \rho_o W_o y_t + \rho_w W_w y_t + X_t\beta + \varepsilon_t$$

### SARFlowSeparablePanel

$$y_t = \rho_d W_d y_t + \rho_o W_o y_t - \rho_d \rho_o W_w y_t + X_t\beta + \varepsilon_t$$

### PoissonSARFlowPanel

$$y_{ij,t} \sim \operatorname{Poisson}(\lambda_{ij,t}), \quad \log \boldsymbol{\lambda}_t = A(\boldsymbol{\rho})^{-1} X_t\beta$$

### PoissonSARFlowSeparablePanel

$$y_{ij,t} \sim \operatorname{Poisson}(\lambda_{ij,t}), \quad \log \boldsymbol{\lambda}_t = A(\boldsymbol{\rho})^{-1} X_t\beta, \quad \rho_w = -\rho_d \rho_o$$

### SEMFlowPanel

$$y_t = X_t\beta + u_t, \quad u_t = \lambda_d W_d u_t + \lambda_o W_o u_t + \lambda_w W_w u_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma^2 I_N)$$

### SEMFlowSeparablePanel

$$y_t = X_t\beta + u_t, \quad u_t = \lambda_d W_d u_t + \lambda_o W_o u_t - \lambda_d \lambda_o W_w u_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma^2 I_N)$$
