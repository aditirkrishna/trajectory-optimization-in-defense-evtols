# Efficiency Maps Dataset Documentation

## Purpose
This dataset defines **rotor efficiency characteristics** across different operating conditions for defense eVTOL propulsion systems. It captures the relationship between rotor speed, thrust output, electrical power input, and efficiency for trajectory optimization and energy management.

**Applications:**
- Energy consumption prediction
- Optimal rotor speed selection
- Power budget allocation
- Mission endurance calculation
- Trajectory optimization constraints

---

## Dataset Schema
```
rotor_id,rpm,thrust_N,power_W,efficiency_percent,notes
```

---

## Specifications

| Parameter | Type | Unit | Description |
|-----------|------|------|-------------|
| rotor_id | string | - | Unique rotor identifier |
| rpm | integer | RPM | Rotor rotational speed |
| thrust_N | float | N | Thrust output force |
| power_W | float | W | Electrical power input |
| efficiency_percent | float | % | Thrust-to-power efficiency |
| notes | string | - | Test conditions and source |

---

## Rotor Types Covered

### 1. Main Rotors (main_rotor_1, main_rotor_2)
- **Purpose**: Primary lift generation
- **RPM Range**: 800-1200 RPM
- **Thrust Range**: 1150-2400 N
- **Efficiency**: 84-89%
- **Configuration**: Symmetric pair for balanced lift

### 2. Tail Rotor (tail_rotor_1)
- **Purpose**: Yaw control and anti-torque
- **RPM Range**: 1200-1800 RPM
- **Thrust Range**: 200-350 N
- **Efficiency**: 78-82%
- **Configuration**: Smaller diameter, higher RPM

### 3. Lift Fan (lift_fan_1)
- **Purpose**: Ducted lift augmentation
- **RPM Range**: 600-900 RPM
- **Thrust Range**: 800-1250 N
- **Efficiency**: 88-90%
- **Configuration**: Ducted fan for enhanced efficiency

### 4. Propeller (propeller_1)
- **Purpose**: Forward flight propulsion
- **RPM Range**: 1500-2400 RPM
- **Thrust Range**: 500-800 N
- **Efficiency**: 83-86%
- **Configuration**: High-speed cruise optimization

---

## Mathematical Models

### 1. Efficiency Definition
$$
\eta = \frac{T \cdot v_{induced}}{P_{electrical}} \times 100\%
$$

Where:
- $T$ = Thrust (N)
- $v_{induced}$ = Induced velocity (m/s)
- $P_{electrical}$ = Electrical power input (W)

### 2. Thrust Scaling (Momentum Theory)
$$
T = 2\rho A v_{induced}^2
$$

Where:
- $\rho$ = Air density (kg/m³)
- $A$ = Rotor disk area (m²)
- $v_{induced}$ = Induced velocity (m/s)

### 3. Power Scaling
$$
P_{induced} = T \cdot v_{induced}
$$
$$
P_{profile} = \frac{1}{8} \rho A C_d \Omega^3 R^3
$$

Where:
- $P_{induced}$ = Induced power
- $P_{profile}$ = Profile power
- $C_d$ = Profile drag coefficient
- $\Omega$ = Angular velocity (rad/s)
- $R$ = Rotor radius (m)

### 4. Efficiency Peak Characteristics
- **Peak Efficiency**: Typically occurs at 70-80% of maximum RPM
- **Efficiency Drop**: Decreases at very low and very high RPM
- **Power Scaling**: Approximately cubic with RPM
- **Thrust Scaling**: Approximately quadratic with RPM

---

## Variable Glossary

| Variable | Meaning | Unit | Typical Range | Notes |
|----------|---------|------|---------------|-------|
| **rotor_id** | Rotor identifier | - | main_rotor_1, tail_rotor_1, etc. | Unique per rotor type |
| **rpm** | Rotational speed | RPM | 600-2400 | Varies by rotor type |
| **thrust_N** | Thrust output | N | 200-2400 | Function of RPM and rotor design |
| **power_W** | Electrical power | W | 8000-78000 | Includes motor and ESC losses |
| **efficiency_percent** | Thrust-to-power efficiency | % | 78-90 | Peak at optimal RPM |
| **notes** | Test conditions | - | - | Source and operating mode |

---

## Efficiency Characteristics

### 1. Main Rotors
- **Peak Efficiency**: ~88.5% at 1000 RPM
- **Operating Range**: 800-1200 RPM
- **Efficiency Drop**: ~2-3% at extremes
- **Power Scaling**: ~1.8x from min to max RPM

### 2. Tail Rotor
- **Peak Efficiency**: ~82% at 1800 RPM
- **Operating Range**: 1200-1800 RPM
- **Lower Efficiency**: Due to smaller diameter and higher RPM
- **Yaw Authority**: Increases with RPM

### 3. Lift Fan (Ducted)
- **Peak Efficiency**: ~90% at 700 RPM
- **Duct Advantage**: 5-10% efficiency improvement
- **Operating Range**: 600-900 RPM
- **Thrust Density**: Higher than open rotors

### 4. Propeller
- **Peak Efficiency**: ~86% at 2400 RPM
- **Forward Flight**: Optimized for cruise
- **Operating Range**: 1500-2400 RPM
- **Speed Optimization**: Higher RPM for cruise efficiency

---

## Usage in Trajectory Optimization

### 1. Power Prediction
```python
# Interpolate power for given thrust requirement
power_required = interpolate_power(thrust_needed, rpm_target, rotor_id)
```

### 2. Efficiency Optimization
```python
# Find optimal RPM for maximum efficiency
optimal_rpm = find_peak_efficiency(rotor_id)
```

### 3. Energy Integration
```python
# Calculate energy consumption over time
energy_consumed = integrate_power_over_time(power_curve, time_interval)
```

### 4. Constraint Handling
```python
# Check rotor limits
thrust_limit = max_thrust_for_rotor(rotor_id, rpm_current)
power_limit = max_power_for_rotor(rotor_id, rpm_current)
```

---

## Data Quality & Validation

### 1. Physical Consistency Checks
- **Power > 0**: All power values positive
- **Thrust > 0**: All thrust values positive
- **Efficiency Bounds**: 0% < efficiency < 100%
- **Monotonic Thrust**: Thrust increases with RPM
- **Monotonic Power**: Power increases with RPM

### 2. Efficiency Validation
- **Peak Efficiency**: Each rotor has clear efficiency peak
- **Realistic Values**: Efficiency within typical rotor ranges
- **Consistent Scaling**: Power and thrust follow expected trends

### 3. Cross-Rotor Validation
- **Main Rotor Symmetry**: main_rotor_1 and main_rotor_2 similar
- **Size Scaling**: Larger rotors have higher thrust/power
- **Type Differences**: Different rotor types show expected characteristics

---

## Integration with Other Datasets

### 1. Battery Integration
```python
# Connect rotor power to battery consumption
battery_power = sum(rotor_powers) / battery_efficiency
```

### 2. Mass Integration
```python
# Update vehicle mass with rotor configurations
total_mass = base_mass + sum(rotor_masses)
```

### 3. Aerodynamic Integration
```python
# Combine with airfoil data for complete propulsion model
total_thrust = sum(rotor_thrusts) + aerodynamic_forces
```

---

## Extensions & Future Work

### 1. Environmental Effects
- **Altitude Effects**: Efficiency changes with air density
- **Temperature Effects**: Motor efficiency variations
- **Wind Effects**: Crosswind impact on rotor performance

### 2. Dynamic Effects
- **Transient Response**: RPM change dynamics
- **Vibration Effects**: Structural coupling
- **Thermal Effects**: Motor heating and derating

### 3. Advanced Models
- **CFD Integration**: High-fidelity aerodynamic models
- **Motor Models**: Detailed electrical characteristics
- **Control Models**: ESC and motor controller dynamics

---

## Example Usage Scenarios

### 1. Hover Optimization
```python
# Find optimal hover configuration
hover_thrust = vehicle_mass * 9.81  # N
optimal_rpm = optimize_hover_efficiency(hover_thrust, main_rotors)
```

### 2. Transition Analysis
```python
# Analyze VTOL to forward flight transition
transition_power = analyze_transition_power(rotor_configs, flight_path)
```

### 3. Mission Planning
```python
# Calculate mission energy requirements
mission_energy = calculate_mission_energy(mission_profile, efficiency_maps)
```

---

## Validation & Testing

### 1. Unit Tests
- Power calculation consistency
- Efficiency bounds checking
- Interpolation accuracy

### 2. Integration Tests
- Battery integration validation
- Trajectory optimization convergence
- Energy conservation checks

### 3. Performance Tests
- Computational efficiency
- Memory usage optimization
- Real-time capability assessment

---

## References

1. Leishman, J. G. (2006). *Principles of Helicopter Aerodynamics*. Cambridge University Press.
2. Johnson, W. (1980). *Helicopter Theory*. Princeton University Press.
3. Stepniewski, W. Z., & Keys, C. N. (1984). *Rotary-Wing Aerodynamics*. Dover Publications.
