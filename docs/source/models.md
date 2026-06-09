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

### NegativeBinomialFlow

$$y_{ij} \sim \operatorname{NegBin}(\mu_{ij}, \alpha), \quad \log \boldsymbol{\mu} = X\beta$$

### SARFlow

$$y = \rho_d W_d y + \rho_o W_o y + \rho_w W_w y + X\beta + \varepsilon$$

### SARFlowSeparable

$$y = \rho_d W_d y + \rho_o W_o y - \rho_d \rho_o W_w y + X\beta + \varepsilon$$

### NegativeBinomialSARFlow

$$y_{ij} \sim \operatorname{NegBin}(\mu_{ij}, \alpha), \quad \log \boldsymbol{\mu} = A(\boldsymbol{\rho})^{-1} X\beta$$

### NegativeBinomialSARFlowSeparable

$$y_{ij} \sim \operatorname{NegBin}(\mu_{ij}, \alpha), \quad \log \boldsymbol{\mu} = A(\boldsymbol{\rho})^{-1} X\beta, \quad \rho_w = -\rho_d \rho_o$$

### SEMFlow

$$y = X\beta + u, \quad u = \lambda_d W_d u + \lambda_o W_o u + \lambda_w W_w u + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2 I)$$

### SEMFlowSeparable

$$y = X\beta + u, \quad u = \lambda_d W_d u + \lambda_o W_o u - \lambda_d \lambda_o W_w u + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2 I)$$

## Panel Flow Models

Stack the flow models above across $T$ periods in time-first order. The NB panel variants currently operate in pooled mode.

### OLSFlowPanel

$$y_t = X_t\beta + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma^2 I_N)$$

### NegativeBinomialFlowPanel

$$y_{ij,t} \sim \operatorname{NegBin}(\mu_{ij,t}, \alpha), \quad \log \boldsymbol{\mu}_t = X_t\beta$$

### SARFlowPanel

$$y_t = \rho_d W_d y_t + \rho_o W_o y_t + \rho_w W_w y_t + X_t\beta + \varepsilon_t$$

### SARFlowSeparablePanel

$$y_t = \rho_d W_d y_t + \rho_o W_o y_t - \rho_d \rho_o W_w y_t + X_t\beta + \varepsilon_t$$

### NegativeBinomialSARFlowPanel

$$y_{ij,t} \sim \operatorname{NegBin}(\mu_{ij,t}, \alpha), \quad \log \boldsymbol{\mu}_t = A(\boldsymbol{\rho})^{-1} X_t\beta$$

### NegativeBinomialSARFlowSeparablePanel

$$y_{ij,t} \sim \operatorname{NegBin}(\mu_{ij,t}, \alpha), \quad \log \boldsymbol{\mu}_t = A(\boldsymbol{\rho})^{-1} X_t\beta, \quad \rho_w = -\rho_d \rho_o$$

### SEMFlowPanel

$$y_t = X_t\beta + u_t, \quad u_t = \lambda_d W_d u_t + \lambda_o W_o u_t + \lambda_w W_w u_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma^2 I_N)$$

### SEMFlowSeparablePanel

$$y_t = X_t\beta + u_t, \quad u_t = \lambda_d W_d u_t + \lambda_o W_o u_t - \lambda_d \lambda_o W_w u_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma^2 I_N)$$

## Sampling Backends

### NUTS (default)

All models support NUTS (No-U-Turn Sampler) via PyMC by default. NUTS handles arbitrary posterior geometries but can be slow for spatial models due to the banana-shaped posterior created by the spatial Jacobian.

### Gibbs Sampler (Gaussian models)

Gaussian cross-sectional models (SAR, SEM, SDM, SDEM) support a custom block-Gibbs sampler via `sampler="gibbs"`:

```python
model = SAR(y=y, X=X, W=W)
idata = model.fit(sampler="gibbs", draws=2000, tune=1000, chains=4)
```

The Gibbs sampler exploits conditional conjugacy with a 3-block strategy:

| Block | Full conditional | Update |
|---|---|---|
| β \| ρ, σ², y | Normal | Direct draw (conjugate) |
| σ² \| β, ρ, y | Inverse-Gamma | Direct draw (conjugate) |
| ρ/λ \| β, σ², y | 1-D non-conjugate | Slice sampling or MALA |

Two execution backends are available:

- **NumPy** (`gibbs_method="numpy"`, default): Adaptive slice sampling for ρ/λ. Pure Python/NumPy, no JAX dependency.
- **JAX** (`gibbs_method="jax"`): Full-JIT compilation via `@eqx.filter_jit`. Uses MALA (gradient-guided) or RW-MH for ρ/λ. Requires JAX and equinox.

See the [Gibbs Sampler User Guide](user-guide/gibbs_sampler.ipynb) for details.

### Gibbs Sampler (NB flow models)

NB flow models (`NegativeBinomialSARFlow`, `NegativeBinomialSARFlowSeparable`, `NegativeBinomialFlow`) support a Pólya–Gamma Gibbs sampler via `sampler="gibbs"`:

```python
model = NegativeBinomialSARFlow(y=y_int, G=G, X=X)
idata = model.fit(sampler="gibbs", draws=2000, tune=1000, chains=4)
```

The sampler uses a reduced-form Pólya–Gamma augmentation strategy with no σ² parameter — spatial dependence enters only through the mean propagator $A^{-1}$:

| Block | Full conditional | Update |
|---|---|---|
| ω \| β, α, y | Pólya–Gamma | Direct draw (conjugate augmentation) |
| β \| ρ, ω, y | Normal | Direct draw (conjugate, via $\tilde{X} = A^{-1}X$) |
| ρ \| ω, y | 1-D non-conjugate | Adaptive slice sampling (β marginalised) |
| α \| y, η | 1-D non-conjugate | Slice sampling on log(α) |

For the unrestricted model (`NegativeBinomialSARFlow`), each ρ parameter (ρ_d, ρ_o, ρ_w) is updated via independent 1-D slice sampling with β marginalised out. For the separable model (`NegativeBinomialSARFlowSeparable`), ρ_w = −ρ_d·ρ_o is deterministic and only ρ_d and ρ_o are sampled. The aspatial `NegativeBinomialFlow` omits the ρ block entirely.
