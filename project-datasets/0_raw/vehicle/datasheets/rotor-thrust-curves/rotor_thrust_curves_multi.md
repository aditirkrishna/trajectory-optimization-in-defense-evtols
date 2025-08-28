# Rotor Thrust Curves Dataset Documentation

## Purpose
This dataset defines the **propulsive characteristics** of the eVTOL's rotors across a range of diameters, altitudes, temperatures, and RPMs.  
It is essential for:
- Modeling thrust generation,
- Energy consumption estimation,
- Efficiency evaluation,
- Trajectory feasibility analysis.

---

## Dataset Schema
```
rotor_diameter_m,altitude_m,temp_offset_C,rpm,thrust_N,torque_Nm,efficiency_percent
```

---

## Specifications
- **Rotor Diameters**: 2.0 m, 2.5 m, 3.0 m  
- **Altitude Levels**: 0 m, 2000 m, 4000 m  
- **Temperature Offsets**: -20 °C, 0 °C, +20 °C (relative to ISA)  
- **RPM Range**: 1000 → 6000 (step 100)  

The dataset includes **thrust, torque, and efficiency** under each condition.

---

## Assumptions
1. **Aerodynamic Model**
   - Rotor modeled with non-dimensional coefficients:
     - Thrust coefficient: $ C_T = 0.12 $
- Torque coefficient: $ C_Q = 0.05 $
   - Assumed constant across operating conditions (first-order approximation).

2. **Air Density**
   - Based on International Standard Atmosphere (ISA).
   - Adjusted for altitude (exponential decay) and temperature offset.

3. **Efficiency**
   - Modeled with a **parabolic efficiency curve**:
     - Peak efficiency = 80% at ~3500 RPM
     - Decreases at very low or high RPM.

4. **Multi-condition Coverage**
   - Dataset expanded to cover varying diameters, altitude, and temperature for ML training.
   - This prevents overfitting to single-condition lookups.

---

## Mathematical Derivations

### 1. Thrust
$$
T = C_T \cdot \rho \cdot n^2 \cdot D^4
$$

Where:  
- $ T $: Thrust [N]  
- $ C_T $: Thrust coefficient (0.12)  
- $ \rho $: Air density [kg/m³]  
- $ n $: Rotational speed [rev/s]  
- $ D $: Rotor diameter [m]  

---

### 2. Torque
$$
Q = C_Q \cdot \rho \cdot n^2 \cdot D^5
$$

Where:  
- $ Q $: Torque [N·m]  
- $ C_Q $: Torque coefficient (0.05)  

---

### 3. Efficiency
$$
\eta(rpm) = \eta_{max} - k \cdot (rpm - rpm_{opt})^2
$$

Where:  
- $ \eta_{max} = 0.80 $  
- $ rpm_{opt} = 3500 $  
- $ k $: small quadratic penalty constant (~1e-7)

---

## Variable Glossary

| Variable | Meaning | Unit | Typical Range |
|----------|---------|------|---------------|
| **rotor_diameter_m** | Rotor diameter | m | 2.0 – 3.0 |
| **altitude_m** | Operating altitude | m | 0 – 4000 |
| **temp_offset_C** | Temperature deviation from ISA | °C | -20, 0, +20 |
| **rpm** | Rotor speed | rev/min | 1000 – 6000 |
| **thrust_N** | Generated thrust | Newtons | 200 – 7000 N (per rotor) |
| **torque_Nm** | Torque demand | N·m | 50 – 2000 |
| **efficiency_percent** | Propulsive efficiency | % | 50 – 80 |
| **source** | Dataset origin | string | "synthetic_multi_condition" |

---

## Usage Notes
- **Trajectory Optimization**: Thrust vs. RPM curve is used to determine climb performance.  
- **Energy Consumption**: Torque × RPM gives mechanical power draw → feeds into battery model.  
- **Condition Variability**: Multiple altitudes & temperatures allow ML models to generalize.  
- **Scaling**: Efficiency assumptions can be tuned with rotor CFD or wind tunnel data later.  

---

## Validation & Next Steps
- Current dataset = **synthetic baseline**.  
- Later refinements:
  - CFD analysis of blade geometry,
  - Wind-tunnel test data,
  - Noise & vibration modeling (not yet included).

---

## Dataset Statistics

### Coverage Matrix
- **Total Records**: 1,379 data points
- **Rotor Diameters**: 3 variants (2.0m, 2.5m, 3.0m)
- **Altitude Levels**: 3 variants (0m, 2000m, 4000m)
- **Temperature Offsets**: 3 variants (-20°C, 0°C, +20°C)
- **RPM Steps**: 51 steps (1000-6000 RPM, 100 RPM increments)

### Performance Ranges
| Diameter | Max Thrust (N) | Max Torque (N·m) | Peak Efficiency (%) |
|----------|----------------|------------------|-------------------|
| 2.0m | ~23,500 | ~19,600 | 80.0 |
| 2.5m | ~57,400 | ~59,800 | 80.0 |
| 3.0m | ~79,600 | ~99,500 | 80.0 |

### Environmental Effects
- **Altitude Impact**: Thrust decreases by ~30% from 0m to 4000m
- **Temperature Impact**: Cold air (-20°C) provides ~0.25% more thrust than hot air (+20°C)
- **Density Scaling**: Follows ISA model with temperature corrections

---

## Implementation Notes

### Data Generation
The dataset was generated using a physics-based model that:
1. Calculates air density using ISA + temperature offset
2. Applies thrust/torque coefficients to rotor geometry
3. Models efficiency as a parabolic function of RPM
4. Scales results across all diameter/altitude/temperature combinations

### Interpolation
For values between data points:
- **Linear interpolation** recommended for RPM
- **Logarithmic interpolation** for altitude effects
- **Linear interpolation** for temperature effects

### Extrapolation Limits
- **RPM**: Not recommended beyond 1000-6000 range
- **Altitude**: Valid up to 4000m (beyond requires different models)
- **Temperature**: Valid for ±20°C offset from ISA

---

## Related Datasets
- `mass_inertia.csv`: Vehicle mass properties for dynamics
- `battery_specs.csv`: Energy storage characteristics
- `limits.csv`: Operational constraints and safety margins

---

## References
- International Standard Atmosphere (ISA) model
- Blade Element Momentum Theory (BEMT)
- Rotorcraft aerodynamics fundamentals
- eVTOL propulsion system design guidelines
