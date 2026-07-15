# Radial-Based Structural Verification Framework for Tropical Cyclones

This framework outlines a physically consistent, radial-verification methodology for evaluating tropical cyclone (TC) structure. By mapping surface wind, pressure, and precipitation fields into a 1D azimuthal-radial coordinate space $(r)$, this framework isolates structural and dynamical errors while bypassing the dual-penalty issues of standard grid-to-grid metrics (such as spatial RMSE).

---

## 1. Core Mathematical Pre-Processing

To implement these metrics, all 2D gridded fields (Latitude, Longitude) must be transformed into a storm-centered cylindrical coordinate system $(r, \theta)$ relative to the observed or simulated center position at a given time step:

1. **Center Identification:** Establish the storm eye coordinates $(X_c, Y_c)$ using minimum pressure or maximum relative vorticity.
2. **Distance Calculation:** Compute the great-circle radial distance ($r$) from the center to every grid point.
3. **Spatial Binning:** Segment data into uniform radial rings (e.g., $\Delta r = 25\text{ km}$ bins for 25km data) extending from $r = 0$ to $r = 1200\text{ km}$. but dropping bins w/ average windspeed less than 10 m/s
4. **Statistical Decomposition:** For each radial bin, compute both the **Azimuthal Mean** ($\bar{X}(r)$) and the **Azimuthal Variance** ($\sigma^2_X(r)$).

---

## 2. Framework Error Metrics

### A. Surface Wind Structure & Intensity Metrics
Wind verification isolates inner-core scale size errors from absolute peak intensity errors.

*   **Radius of Maximum Winds Error ($\Delta R_{max}$):**
    $$\Delta R_{max} = \left| R_{max,\text{ sim}} - R_{max,\text{ obs}} \right|$$
    *Measures inner-core size, eyewall localization, and structural compaction.*
*   **Critical Radii Extent Error ($\Delta R_{34}, \Delta R_{50}, \Delta R_{64}$):**
    The difference in radial distance where the azimuthally averaged wind speed drops below standard operational thresholds ($34\text{ kt}$, $50\text{ kt}$, and $64\text{ kt}$) [convert to m/s].
    $$\Delta R_{\text{threshold}} = R_{\text{threshold, sim}} - R_{\text{threshold, obs}}$$
    *Measures the error in the broadness of the outer wind envelope (gale-force vs. hurricane-force extents).*
*   **Integrated Wind Profile Mean Absolute Error ($MAE_{v\text{\_radial}}$):**
    $$MAE_{v\text{\_radial}} = \frac{1}{N} \sum_{i=1}^{N} \left| \bar{V}_{\text{sim}}(r_i) - \bar{V}_{\text{obs}}(r_i) \right|$$
    *Evaluates the complete shape envelope error of the wind profile from core to environment.*

### B. Mass-Momentum & Dynamic Consistency Metrics
These metrics evaluate whether the simulated fields obey the governing physical laws of tropical cyclone boundary layers (e.g., Gradient Wind Balance).

*   **Pressure Gradient Force ($PGF$) Profile:**
    Compute the derivative of the azimuthally averaged radial pressure profile:
    $$PGF(r) = \frac{1}{\rho}\frac{\partial \bar{P}}{\partial r}$$
*   **Radial Distance Mismatch ($\Delta R_{\text{mismatch}}$):**
    Identify the radius of maximum pressure gradient ($R_{PGFmax}$) and the radius of maximum wind ($R_{Vmax}$).
    $$\Delta R_{\text{mismatch}} = \left| (R_{PGFmax,\text{ sim}} - R_{Vmax,\text{ sim}}) - (R_{PGFmax,\text{ obs}} - R_{Vmax,\text{ obs}}) \right|$$
    *Measures physical alignment. In balanced mature storms, $R_{PGFmax}$ should tightly lead or align inside $R_{Vmax}$. A large mismatch indicates severe boundary layer or friction parametrization errors.*
*   **Integrated Wind-Pressure Deficit Deviation ($\Delta E_{\text{gwb}}$):**
    Evaluates the cyclostrophic imbalance by comparing the actual central pressure deficit ($\Delta P = P_{\text{outer}} - P_{\text{center}}$) against the wind-derived expected profile:
    $$\text{Imbalance Error} = \left| \left[ \Delta P - \int_{0}^{R_{\text{outer}}} \frac{\bar{V}(r)^2}{r} dr \right]_{\text{sim}} - \left[ \Delta P - \int_{0}^{R_{\text{outer}}} \frac{\bar{V}(r)^2}{r} dr \right]_{\text{obs}} \right|$$

### C. Radial Precipitation Morphology Metrics
Isolates core convective rain rates from the spatial placement of rainbands.

*   **Eyewall Peak Location and Intensity Errors ($\Delta R_m, \Delta T_m$):**
    From the radial precipitation profile, identify the maximum rain rate ($T_m$) and the radius at which it occurs ($R_m$).
    $$\Delta R_m = \left| R_{m,\text{ sim}} - R_{m,\text{ obs}} \right| \quad \text{and} \quad \Delta T_m = T_{m,\text{ sim}} - T_{m,\text{ obs}}$$
    *Tracks eyewall precipitation displacement and localized thermodynamic intensity.*
*   **Moisture Envelope Decay Error ($\Delta R_e$):**
    Calculate the e-folding radius ($R_e$), defined as the radial distance outward from $R_m$ where the rain rate drops to $1/e$ ($\approx 37\%$) of the peak eyewall intensity ($T_m$).
    $$\Delta R_e = \left| R_{e,\text{ sim}} - R_{e,\text{ obs}} \right|$$
    *Differentiates tight, intense convective storms from wide, stratiform-heavy systems.*

### D. Spatial Structure & Symmetry Metrics (Variance Analysis)
Variance metrics evaluate whether a storm is highly asymmetric (e.g., sheared/unbalanced) or perfectly circular.

*   **Azimuthal Variance Profile Error ($MAE_{\sigma\text{\_radial}}$):**
    For each radial bin $r_i$, compute the variance of the field around the azimuth. Compute the MAE between the simulated and observed variance curves:
    $$MAE_{\sigma\text{\_radial}} = \frac{1}{N} \sum_{i=1}^{N} \left| \sigma^2_{\text{sim}}(r_i) - \sigma^2_{\text{obs}}(r_i) \right|$$
    *Measures whether the model correctly captures storm asymmetry (e.g., outer rainbands, open eyewalls, wave-1 asymmetries).*
*   **Radial Gradient Sharpness ($\text{Var}_{\text{profile}}$):**
    Compute the total variance across the 1D mean radial profile arrays themselves ($\text{Var}(\bar{X})$).
    $$\Delta \text{Var}_{\text{profile}} = \text{Var}(\bar{X}_{\text{sim}}) - \text{Var}(\bar{X}_{\text{obs}})$$
    *Measures gradient sharpness. A positive value means the model produces an overly sharp, idealized eye-to-eyewall transition; a negative value means the model's structural features are too diffuse and washed out.*

---

## 3. Verification Scorecard Summary Table

| Metric | Field Focus | Physical Structure Captured | Ideal Value |
| :--- | :--- | :--- | :--- |
| **$\Delta R_{max}$** | Wind | Inner-core eyewall scaling and compactness | $0\text{ km}$ |
| **$\Delta R_{34,50,64}$** | Wind | Broadness of outer destructive wind fields | $0\text{ km}$ |
| **$\Delta R_{\text{mismatch}}$** | Wind / Pressure | Mass-momentum coupling and boundary layer balance | $0\text{ km}$ |
| **$\Delta R_m$** | Precipitation | Convective eyewall localization | $0\text{ km}$ |
| **$\Delta R_e$** | Precipitation | Outer stratiform moisture footprint extent | $0\text{ km}$ |
| **Profile MAE** | All Fields | Total continuous radial shape mismatch | Minimize |
| **$MAE_{\sigma\text{\_radial}}$** | All Fields | Structural asymmetry, shear effects, and rainband organization | Minimize |
| **$\Delta \text{Var}_{\text{profile}}$** | All Fields | Sharpness vs. smoothness of core transitions | $0$ |


## Dev notes 
we will be generating these for an ensemble of generated storms (ensemble, lat, lon) or (time, ensemble, lat, lon) against a target, which may also be an ensemble but for the start will be a true target from the original simulation
for coarse data (100km) the storm is not really resolved so these measures will not be useful. 25km maybe more useful especially for larger storms, but still there will be a large disparity between inner radii and outer radii in the number of samples that go into the measure, so I expect higher errors there maybe?
