# GEK Methods in `mdolab.py` (Beginner-Friendly Guide)

This document explains the online GEK correction methods implemented in:

- `flowvae/app/mdolab.py`
- class: `_OnlineGEKDeltaModel`
- used by `SolverCombined` modes: `10`, `11`, `12`

---

## 1. Why do we need GEK here?

In this workflow, you have:

- a **fast** model (ML surrogate),
- a **high-fidelity but expensive** model (CFD).

The fast model is not exact. So instead of replacing it, we learn the **error** between CFD and ML:

\[
\Delta f(x)=f_{\text{CFD}}(x)-f_{\text{ML}}(x).
\]

Then we predict this error and add it back:

\[
f_{\text{corrected}}(x)=f_{\text{ML}}(x)+\widehat{\Delta f}(x).
\]

So GEK is used as an **online correction model**.

---

## 2. What does “Gradient-Enhanced Kriging” mean?

Normal GP/Kriging uses only function values.  
GEK uses both:

1. function values \(\Delta f(x)\),
2. gradients \(\nabla_x \Delta f(x)\).

Using gradients gives much stronger local information, which is especially useful when CFD samples are few.

In code, gradients come from:

\[
\nabla_x \Delta f(x)=\nabla_x f_{\text{CFD}}(x)-\nabla_x f_{\text{ML}}(x).
\]

---

## 3. High-Level Runtime Flow

For modes `10/11/12`, each optimization cycle behaves as:

1. Run ML (`wrap_cruiseFuncs`).
2. If CFD is triggered this iteration:
   - compute CFD functions,
   - cache ML and CFD values at the same \(x\),
   - later in `wrap_cruiseFuncsSens`, compute gradient deltas and train GEK with one new sample.
3. If CFD is not triggered:
   - GEK predicts \(\widehat{\Delta f}\) and \(\widehat{\nabla \Delta f}\),
   - corrections are added to ML outputs and ML sensitivities.

This is an **online loop**: every new CFD point updates GEK.

---

## 4. Notation and Shapes (important for beginners)

- \(x\in\mathbb{R}^d\): flattened design vector.
- \(n\): number of stored GEK samples (sliding window).
- \(d_r\): effective input dimension used by GEK.
  - After pre-selection by `max_dims` (and further reduced by active subspace in mode 12).
- For each output key (`wing_cd`, `wing_cl`, ...), GEK is fit independently.

Training data for one output key:

- function deltas: \(y=[\Delta f_1,\dots,\Delta f_n]\in\mathbb{R}^{n}\),
- gradient deltas: \(G=[g_1;\dots;g_n]\in\mathbb{R}^{n\times d_r}\), where \(g_i=\nabla_x\Delta f(x_i)\) in the model’s working coordinates.

---

## 5. Kernel and GEK Covariance (core equations)

RBF kernel:

\[
k(x_i,x_j)=\exp\!\left(-\frac12\sum_{m=1}^{d_r}\frac{(x_{i,m}-x_{j,m})^2}{\ell_m^2}\right).
\]

Symbol explanation (used in this section):

- \(x_i, x_j\): two sample design vectors used by GEK.
- \(d_r\): effective GEK input dimension (after truncation/projection).
- \(x_{i,m}\): component \(m\) of vector \(x_i\).
- \(\ell_m\): kernel length scale of component \(m\).
- \(k(x_i,x_j)\): RBF similarity between two samples.
- \(r_{ij}\): sample difference vector \(x_i-x_j\).
- \(\Lambda^{-1}\): diagonal matrix with entries \(1/\ell_m^2\).
- \(\delta_{ab}\): Kronecker delta (1 if \(a=b\), else 0).
- \(K_{ff}\): function-value covariance block.
- \(K_{fg}\): function-to-gradient covariance block.
- \(K_{gg}\): gradient-to-gradient covariance block.
- \(\eta\): nugget regularization coefficient.
- \(I\): identity matrix.
- \(y_{\text{obs}}\): stacked observations \([\Delta f,\nabla\Delta f]\).
- \(\alpha\): GEK linear coefficients, solved from \(K\alpha=y_{\text{obs}}\).

Define \(r_{ij}=x_i-x_j\), and \(\Lambda^{-1}=\mathrm{diag}(1/\ell_m^2)\).

### 5.1 Function-function block
\[
K_{ff}[i,j]=k(x_i,x_j).
\]

Intuition:

- This is the same block as standard GP/Kriging.
- It says: if two design points are close, their **correction values** \(\Delta f\) should be similar.
- So \(K_{ff}\) mainly controls value interpolation/smoothing.

### 5.2 Function-gradient block
\[
K_{fg}[i,(j,m)]
=\frac{\partial k(x_i,x_j)}{\partial x_{j,m}}
=k(x_i,x_j)\frac{(x_{i,m}-x_{j,m})}{\ell_m^2}.
\]

Intuition:

- This block couples a **value** at one point with a **slope component** at another point.
- It tells the model how moving in direction \(m\) changes nearby values.
- Without this block, gradient observations cannot influence value prediction correctly.

### 5.3 Gradient-gradient block
\[
K_{gg}[(i,a),(j,b)]
=\frac{\partial^2 k(x_i,x_j)}{\partial x_{i,a}\partial x_{j,b}}
=k(x_i,x_j)\left[\delta_{ab}\frac1{\ell_a^2}
-\frac{(x_{i,a}-x_{j,a})(x_{i,b}-x_{j,b})}{\ell_a^2\ell_b^2}\right].
\]

Intuition:

- This block correlates **slope with slope** (component \(a\) at \(x_i\) with component \(b\) at \(x_j\)).
- It enforces consistency between gradient measurements across samples.
- Practically, it helps the model reconstruct local shape (how fast and in which directions the correction changes).

Full GEK matrix:

\[
K=
\begin{bmatrix}
K_{ff} & K_{fg}\\
K_{fg}^{\top} & K_{gg}
\end{bmatrix}
+\eta I,
\]

where \(\eta\) is `nugget` for numerical stability.

How to understand the full block matrix:

- Top-left \(K_{ff}\): "value-value agreement".
- Top-right \(K_{fg}\): "value-slope coupling".
- Bottom-right \(K_{gg}\): "slope-slope agreement".
- Together they force the fitted correction to match both **function values** and **gradients** at training points.
- In short: standard GP uses only "height" data; GEK uses both "height" and "tilt" data.

Observation vector:

\[
y_{\text{obs}}=
[\Delta f_1,\dots,\Delta f_n,\nabla\Delta f_1,\dots,\nabla\Delta f_n]^\top.
\]

Solve:

\[
\alpha=K^{-1}y_{\text{obs}}
\]

(least-squares fallback is used if direct solve is ill-conditioned).

---

## 6. Prediction Equations

At query \(x_*\), build cross-covariances \(k_*(x_*)\) against all stored samples.

Function correction:

\[
\widehat{\Delta f}(x_*)=k_*(x_*)^\top\alpha.
\]

Gradient correction (for each coordinate \(b\)):

\[
\widehat{\frac{\partial\Delta f}{\partial x_b}}(x_*)
=k_*^{(b)}(x_*)^\top\alpha.
\]

Then corrected outputs passed to optimizer are:

\[
f_{\text{ML}}+\widehat{\Delta f},\qquad
\nabla f_{\text{ML}}+\widehat{\nabla\Delta f}.
\]

---

## 7. Original Kriging / Gaussian Process Regression (GPR)

Before GEK, the baseline surrogate is standard Kriging (also called Gaussian Process Regression).
It uses only function values, without gradient observations.

Kriging vs Gaussian Process (important):

- In many engineering optimization papers/tools, people say **Kriging**.
- In machine learning/statistics, the same core method is called **Gaussian Process Regression (GPR)**.
- For practical use in this project, you can treat them as the same surrogate family:
  - prior over functions + kernel-based covariance + Bayesian update from data.
  - Differences in wording are mostly historical/domain conventions.

What is a kernel? (intuition first):

- A kernel \(k(x_i,x_j)\) is a **similarity function** between two inputs.
- If two points are similar, kernel value is large; if far apart, kernel value is small.
- In GP/Kriging, this similarity is used as covariance:
  - high kernel value \(\Rightarrow\) model expects outputs to move together,
  - low kernel value \(\Rightarrow\) weaker coupling between outputs.
- So the kernel is the "rule of smoothness/correlation" for your surrogate.

RBF kernel intuition:

\[
k(x_i,x_j)=\exp\!\left(-\frac12\sum_m\frac{(x_{i,m}-x_{j,m})^2}{\ell_m^2}\right).
\]

- Distance grows \(\Rightarrow\) exponent becomes more negative \(\Rightarrow\) kernel goes toward 0.
- \(\ell_m\) controls how fast correlation decays in direction \(m\):
  - large \(\ell_m\): slow decay (smoother, long-range correlation),
  - small \(\ell_m\): fast decay (more local behavior).

Given training pairs:

\[
\mathcal{D}=\{(x_i,y_i)\}_{i=1}^{n},\qquad y_i=f(x_i),
\]

define kernel matrix and query covariance vector:

\[
K_{ij}=k(x_i,x_j),\qquad
k_*(x_*)=[k(x_*,x_1),\ldots,k(x_*,x_n)]^\top.
\]

With nugget/noise \(\sigma_n^2\):

\[
\alpha=(K+\sigma_n^2 I)^{-1}y.
\]

Prediction mean:

\[
\hat f(x_*)=k_*(x_*)^\top\alpha.
\]

Prediction variance:

\[
\mathrm{Var}[f(x_*)]
=k(x_*,x_*)-k_*(x_*)^\top (K+\sigma_n^2 I)^{-1}k_*(x_*).
\]

Intuitive understanding:

- Kriging says "nearby points should have similar function values."
- The kernel controls what "nearby" means.
- It predicts both a value and an uncertainty.
- But it does not directly enforce slope matching because gradients are not used.

How GEK extends this:

- Standard Kriging uses only \(K_{ff}\) (value-value block).
- GEK adds \(K_{fg}\) and \(K_{gg}\), so both values and gradients are fitted.
- This is why GEK is usually more data-efficient when reliable gradients are available.

---

## 8. Mode-by-Mode Explanation

## 8.1 Mode 10 (baseline online GEK)

`cfd_include_mode = 10`

- Uses raw reduced variables and deltas.
- Good baseline, but can be unstable in high dimension + very few samples.

---

## 8.2 Mode 11 (stable-normalized GEK)

`cfd_include_mode = 11`

Adds robust normalization and kernel safeguards.

### Input normalization
\[
x'=\frac{x-\mu_x}{s_x},\qquad
s_x=\max(\mathrm{std}(x),\texttt{xscale\_floor}).
\]

### Output normalization (per output key)
\[
z=\frac{\Delta f-\mu_y}{s_y},\qquad
s_y=\max(\mathrm{std}(\Delta f),\texttt{yscale\_floor}).
\]

### Gradient consistency under scaling
\[
\nabla_{x'}z=\nabla_x\Delta f\odot\frac{s_x}{s_y}.
\]

### Length-scale floor
\[
\ell_m\leftarrow\max(\ell_m,\texttt{ell\_floor}).
\]

### Kernel exponent clipping
\[
\xi\leftarrow \mathrm{clip}(\xi,-\texttt{exp\_clip},0),\qquad
k=\exp(\xi).
\]

This prevents underflow (`exp(-very large)` -> 0).

### Back-transform after prediction
\[
\widehat{\Delta f}=\widehat z\,s_y+\mu_y,\qquad
\widehat{\nabla_x\Delta f}
=\widehat{\nabla_{x'}z}\odot\frac{s_y}{s_x}.
\]

---

## 8.3 Mode 12 (active-subspace + normalized GEK)

`cfd_include_mode = 12`

Mode 12 = Mode 11 + gradient-based dimensionality reduction.

### Step A: build gradient covariance
Using normalized gradients:

\[
C\approx\frac{1}{N}G^\top G.
\]

### Step B: eigen decomposition
\[
C=W\Lambda W^\top.
\]

Take leading eigenvectors \(W_k\) by:

- hard cap `active_rank`,
- and energy threshold `active_energy`:
\[
\frac{\sum_{i=1}^k\lambda_i}{\sum_{i=1}^{d_r}\lambda_i}\ge \texttt{active\_energy}.
\]

### Step C: project inputs/gradients
\[
z=x'W_k,\qquad
\nabla_z=\nabla_{x'}W_k.
\]

GEK is fit in low-dimensional \(z\)-space.

### Step D: map gradient back
\[
\nabla_{x'}=\nabla_zW_k^\top,\qquad
\nabla_x=\nabla_{x'}\odot\frac{s_y}{s_x}.
\]

This usually improves stability when \(d_r\) is still large for the available sample count.

---

## 9. “What each parameter means” (practical)

### Shared GEK parameters

- `gek_max_points`: window size of online samples.
  - Larger: more history, heavier solve.
- `gek_max_dims`: max retained DV dims before GEK.
  - Smaller: more stable, but may lose information.
- `gek_nugget`: diagonal regularization.
  - Increase if matrix solves are unstable/noisy.

### Logging

- `gek_log_interval`: print every N prediction calls.

### Mode 11/12 stability parameters

- `gek11_xscale_floor`: minimum input scale.
- `gek11_yscale_floor`: minimum output-delta scale.
- `gek11_ell_floor`: minimum GP length scale.
- `gek11_exp_clip`: clamp for kernel exponent.

### Mode 12 active-subspace parameters

- `gek12_active_rank`: maximum subspace rank.
- `gek12_active_energy`: cumulative eigenvalue energy target.

---

## 10. Recommended Starting Values

If you are new, start with:

- `cfd_include_mode=12`
- `gek_max_points=40~100`
- `gek_max_dims=30~60`
- `gek_nugget=1e-8` (increase to `1e-6` if unstable)
- `gek11_ell_floor=0.2`
- `gek12_active_rank=6~12`
- `gek12_active_energy=0.9~0.98`

---

## 11. Diagnosing Common Failures

### Symptom A: predicted deltas are almost always 0
Possible causes:

- kernel underflow (query too far in scaled space),
- too few samples vs dimension,
- poor scaling.

Actions:

- use mode 11 or 12 (preferred),
- reduce `gek_max_dims`,
- increase `gek11_ell_floor`,
- check active-subspace rank in mode 12.

### Symptom B: unstable / noisy correction
Possible causes:

- covariance matrix conditioning issues,
- gradient noise.

Actions:

- increase `gek_nugget`,
- reduce active rank,
- reduce retained dimensions.

### Symptom C: good fit near CFD points but poor between them
Possible causes:

- extrapolation region too large,
- insufficient CFD coverage.

Actions:

- increase CFD update frequency,
- increase sample window,
- use active subspace to focus on dominant directions.

---

## 12. Minimal Pseudocode

```text
for each optimizer iteration:
    run ML funcs/sens
    if CFD triggered:
        run CFD funcs/sens
        compute delta_f = CFD_f - ML_f
        compute delta_g = CFD_g - ML_g
        GEK.add_sample(x, delta_f, delta_g)
        return CFD values to optimizer
    else:
        pred_delta_f, pred_delta_g = GEK.predict(x)
        return ML + predicted correction
```

---

## 13. Relationship to Code

- math model: `_OnlineGEKDeltaModel`
- runtime integration: `SolverCombined.wrap_cruiseFuncs` and `SolverCombined.wrap_cruiseFuncsSens`
- modes:
  - `10`: baseline GEK
  - `11`: normalized/stable GEK
  - `12`: active-subspace + normalized GEK
