# Airfoil Curves Dataset Documentation

## Purpose
This dataset defines the **aerodynamic characteristics of synthetic eVTOL airfoils**.  
It provides **lift, drag, and moment coefficients (Cl, Cd, Cm)** across a range of:
- Angle of attack (AoA),
- Reynolds numbers,
- Mach numbers,
- Airfoil geometries.

These data are critical for:
- Estimating aerodynamic forces,
- Stability and control analysis,
- Energy and performance modeling.

---

## Dataset Schema
```
airfoil,alpha_deg,Re,M,Cl,Cd,Cm,source
```

---

## Specifications
- **Airfoils**:  
  - `NACA2412_synth` (cambered, moderate lift)  
  - `NACA0015_synth` (symmetric, stable)  
  - `CustomBlended_synth` (defense-optimized blended wing)  

- **Angle of Attack (AoA)**: -10° → +20° (1° step)  
- **Reynolds Numbers (Re)**: 5e5, 1e6, 2e6  
- **Mach Numbers (M)**: 0.1, 0.2, 0.3  
- **Outputs**: Cl (lift), Cd (drag), Cm (moment)

---

## Assumptions
1. **Thin Airfoil Theory** for lift slope:
   - \( \frac{dCl}{d\alpha} \approx 2\pi \) per radian.
   - Adjusted slope for cambered and blended airfoils.

2. **Drag Polar Approximation**:
   - Quadratic form:
     \[
     Cd = Cd_{min} + k (Cl - Cl_{opt})^2
     \]
   - Accounts for induced drag and profile drag.

3. **Moment Coefficient (Cm)**:
   - Fixed for each airfoil type.
   - `NACA2412_synth`: -0.05 (nose-down moment).
   - `NACA0015_synth`: -0.02 (slight nose-down moment).
   - `CustomBlended_synth`: -0.04 (moderate nose-down moment).

4. **Reynolds & Mach Effects**:
   - Higher Re → delayed stall, lower drag.
   - Higher Mach → compressibility drag rise.

---

## Mathematical Derivations

### 1. Lift Coefficient
\[
Cl(\alpha) = Cl_0 + a (\alpha - \alpha_{L=0})
\]

Where:
- \( Cl_0 \): baseline lift at zero AoA,
- \( a \): lift slope (per rad),
- \( \alpha \): angle of attack [deg],
- \( \alpha_{L=0} \): zero-lift AoA.

---

### 2. Drag Coefficient
\[
Cd = Cd_{min} + k (Cl - Cl_{opt})^2
\]

Where:
- \( Cd_{min} \): minimum drag,
- \( k \): induced drag factor,
- \( Cl_{opt} \): lift coefficient at minimum drag.

---

### 3. Moment Coefficient
\[
Cm = Cm_0
\]

Constant for each airfoil.

---

## Variable Glossary

| Variable | Meaning | Unit | Typical Range |
|----------|---------|------|---------------|
| **airfoil** | Identifier of airfoil geometry | string | NACA2412_synth, NACA0015_synth, CustomBlended_synth |
| **alpha_deg** | Angle of attack | deg | -10 → +20 |
| **Re** | Reynolds number (density·velocity·chord / μ) | dimensionless | 5e5 → 2e6 |
| **M** | Mach number (velocity/speed of sound) | dimensionless | 0.1 → 0.3 |
| **Cl** | Lift coefficient | – | -0.98 → +2.47 |
| **Cd** | Drag coefficient | – | 0.018 → 0.206 |
| **Cm** | Moment coefficient about quarter-chord | – | -0.05 → -0.02 |
| **source** | Origin of dataset | string | "synthetic_thin_airfoil_model" |

---

## Dataset Characteristics

### Airfoil-Specific Properties

#### NACA2412_synth
- **Zero-lift AoA**: -2° (cambered airfoil)
- **Lift slope**: ~0.11 per degree (6.3 per radian)
- **Moment coefficient**: -0.05 (nose-down)
- **Cl range**: -0.88 to +2.42

#### NACA0015_synth  
- **Zero-lift AoA**: 0° (symmetric airfoil)
- **Lift slope**: ~0.105 per degree (6.0 per radian)
- **Moment coefficient**: -0.02 (slight nose-down)
- **Cl range**: -0.945 to +2.205

#### CustomBlended_synth
- **Zero-lift AoA**: -1° (slightly cambered)
- **Lift slope**: ~0.115 per degree (6.6 per radian)
- **Moment coefficient**: -0.04 (moderate nose-down)
- **Cl range**: -0.9775 to +2.4725

### Reynolds Number Effects
- **Data shows**: No Reynolds number variation in coefficients
- **Implication**: Thin airfoil theory assumption holds
- **Re range**: 500,000 to 2,000,000

### Mach Number Effects  
- **Data shows**: No Mach number variation in coefficients
- **Implication**: Incompressible flow assumption
- **M range**: 0.1 to 0.3

---

## Usage Notes
- **Force Estimation**:  
  Lift = Cl · 0.5 · ρ · V² · S  
  Drag = Cd · 0.5 · ρ · V² · S  

- **Control Design**:  
  Cm values feed into stability and trim analysis.  

- **Generalization**:  
  Multiple airfoils, Reynolds, and Mach conditions prevent ML models from overfitting.  

- **Limitations**:
  - Synthetic data based on thin airfoil theory
  - No stall behavior modeling
  - No compressibility effects
  - No Reynolds number effects

- **Next Steps**:  
  CFD analysis or wind-tunnel data could replace synthetic values for real-world fidelity.

---

## Data Quality Assessment

### Strengths
- **Comprehensive coverage**: 3 airfoils × 31 AoA × 3 Re × 3 M = 837 data points
- **Consistent methodology**: All data from same synthetic model
- **Well-structured**: Clear CSV format with proper headers
- **Realistic ranges**: AoA, Re, and M values appropriate for eVTOL operations

### Limitations
- **Synthetic nature**: Not experimental or CFD-derived
- **Simplified physics**: No stall, compressibility, or Re effects
- **Limited validation**: No comparison with real airfoil data
- **Fixed moment**: Cm constant across all conditions

### Recommendations
1. **Validation**: Compare with published NACA airfoil data
2. **Enhancement**: Add stall behavior and post-stall characteristics
3. **CFD integration**: Replace synthetic data with computational results
4. **Experimental validation**: Wind tunnel testing for key conditions
