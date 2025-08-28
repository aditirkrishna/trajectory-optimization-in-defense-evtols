# Actuator Faults Dataset Documentation

## Purpose
This dataset defines **actuator fault scenarios** for defense eVTOL systems, enabling fault-tolerant trajectory optimization, failure mode analysis, and robust control system design. It models realistic actuator failures that could occur during flight operations.

**Applications:**
- Fault-tolerant control design
- Failure mode and effects analysis (FMEA)
- Robust trajectory optimization
- Emergency landing planning
- System reliability assessment
- Redundancy requirements analysis

---

## Dataset Schema
```
actuator_id,fault_type,start_time_s,end_time_s,severity,notes
```

---

## Specifications

| Parameter | Type | Unit | Description |
|-----------|------|------|-------------|
| actuator_id | string | - | Unique actuator identifier |
| fault_type | string | - | Type of fault (stuck, drift, noise) |
| start_time_s | float | seconds | Fault initiation time |
| end_time_s | float | seconds | Fault termination time |
| severity | float | - | Fault magnitude (0.0-1.0) |
| notes | string | - | Detailed fault description |

---

## Fault Types

### 1. Stuck Faults
- **Description**: Actuator becomes locked at a specific position
- **Severity Range**: 0.5-1.0
- **Examples**: ESC failure, servo lock, valve stuck
- **Impact**: Complete loss of control authority in affected axis

### 2. Drift Faults
- **Description**: Gradual deviation from commanded position
- **Severity Range**: 0.1-0.5
- **Examples**: Sensor drift, wear-related degradation
- **Impact**: Progressive loss of control precision

### 3. Noise Faults
- **Description**: High-frequency oscillations or electrical noise
- **Severity Range**: 0.3-0.6
- **Examples**: Electrical interference, bearing wear
- **Impact**: Reduced control smoothness and precision

---

## Actuator Categories

### 1. Propulsion Actuators
- **main_rotor_1_esc**: Main rotor electronic speed controller
- **main_rotor_2_esc**: Secondary main rotor ESC
- **tail_rotor_motor**: Anti-torque rotor motor
- **throttle_control**: Engine/rotor power control

### 2. Flight Control Actuators
- **elevator_servo**: Pitch control surface
- **aileron_servo**: Roll control surface
- **rudder_servo**: Yaw control surface
- **pitch_trim**: Pitch trim adjustment
- **roll_trim**: Roll trim adjustment
- **yaw_trim**: Yaw trim adjustment

### 3. Secondary Actuators
- **landing_gear**: Landing gear deployment
- **flap_servo**: High-lift device control
- **spoiler_servo**: Drag device control
- **brake_actuator**: Wheel brake control

### 4. System Actuators
- **fuel_valve**: Fuel flow control
- **cooling_fan**: Thermal management

---

## Severity Levels

### 1. Low Severity (0.1-0.3)
- **Impact**: Minimal performance degradation
- **Recovery**: Automatic compensation possible
- **Examples**: Minor drift, low-level noise
- **Response**: Monitor and log

### 2. Medium Severity (0.4-0.7)
- **Impact**: Noticeable performance degradation
- **Recovery**: Requires control law adaptation
- **Examples**: Moderate drift, significant noise
- **Response**: Activate fault-tolerant control

### 3. High Severity (0.8-1.0)
- **Impact**: Severe performance degradation
- **Recovery**: Emergency procedures required
- **Examples**: Complete failure, stuck actuator
- **Response**: Emergency landing or mission abort

---

## Mathematical Models

### 1. Stuck Fault Model
\[
u(t) = u_{stuck} \quad \text{for} \quad t \in [t_{start}, t_{end}]
\]

Where:
- \(u(t)\) = Actuator output
- \(u_{stuck}\) = Stuck position value
- \(t_{start}, t_{end}\) = Fault duration

### 2. Drift Fault Model
\[
u(t) = u_{cmd}(t) + \alpha \cdot (t - t_{start}) \quad \text{for} \quad t \in [t_{start}, t_{end}]
\]

Where:
- \(u_{cmd}(t)\) = Commanded input
- \(\alpha\) = Drift rate (function of severity)
- \(t_{start}\) = Fault initiation time

### 3. Noise Fault Model
\[
u(t) = u_{cmd}(t) + \beta \cdot \sin(\omega t + \phi) \quad \text{for} \quad t \in [t_{start}, t_{end}]
\]

Where:
- \(\beta\) = Noise amplitude (function of severity)
- \(\omega\) = Noise frequency
- \(\phi\) = Random phase

---

## Variable Glossary

| Variable | Meaning | Unit | Range | Notes |
|----------|---------|------|-------|-------|
| **actuator_id** | Actuator identifier | - | - | Unique per actuator |
| **fault_type** | Type of fault | - | stuck, drift, noise | Categorical variable |
| **start_time_s** | Fault start time | seconds | 0-600 | Mission timeline |
| **end_time_s** | Fault end time | seconds | 0-600 | Must > start_time |
| **severity** | Fault magnitude | - | 0.0-1.0 | Normalized scale |
| **notes** | Fault description | - | - | Detailed explanation |

---

## Fault Injection in Simulation

### 1. Time-Based Injection
```python
def inject_fault(actuator_id, fault_type, start_time, end_time, severity):
    if current_time >= start_time and current_time <= end_time:
        return apply_fault_model(actuator_id, fault_type, severity)
    return normal_operation(actuator_id)
```

### 2. Severity Scaling
```python
def scale_fault_effect(base_effect, severity):
    return base_effect * severity
```

### 3. Fault Propagation
```python
def propagate_fault(actuator_fault, system_state):
    # Calculate impact on vehicle dynamics
    return updated_system_state
```

---

## Fault Detection & Diagnosis

### 1. Detection Methods
- **Residual Analysis**: Compare expected vs. actual actuator response
- **Statistical Methods**: Monitor actuator behavior statistics
- **Model-Based**: Use system models for fault detection
- **Signal Processing**: Analyze actuator signals for anomalies

### 2. Diagnosis Algorithms
- **Fault Isolation**: Identify which actuator is faulty
- **Fault Classification**: Determine fault type
- **Severity Estimation**: Quantify fault magnitude
- **Prognosis**: Predict fault evolution

### 3. Response Strategies
- **Passive Fault Tolerance**: Robust control design
- **Active Fault Tolerance**: Control law reconfiguration
- **Redundancy Management**: Switch to backup systems
- **Emergency Procedures**: Safe landing protocols

---

## Integration with Trajectory Optimization

### 1. Fault-Aware Planning
```python
def optimize_trajectory_with_faults(fault_scenarios):
    for fault in fault_scenarios:
        trajectory = optimize_robust_trajectory(fault)
        evaluate_trajectory_safety(trajectory, fault)
```

### 2. Robustness Constraints
```python
def add_fault_tolerance_constraints(optimization_problem):
    # Add constraints for fault scenarios
    for fault in actuator_faults:
        add_fault_constraint(optimization_problem, fault)
```

### 3. Safety Margins
```python
def calculate_safety_margins(actuator_faults):
    # Determine required control margins
    return safety_margins
```

---

## Validation & Testing

### 1. Fault Scenario Validation
- ✅ **Time Consistency**: end_time > start_time
- ✅ **Severity Bounds**: 0.0 ≤ severity ≤ 1.0
- ✅ **Actuator Existence**: Valid actuator_id references
- ✅ **Fault Type Validity**: Recognized fault types

### 2. Simulation Testing
- **Fault Injection**: Verify fault models work correctly
- **System Response**: Test vehicle response to faults
- **Recovery Procedures**: Validate fault recovery
- **Performance Degradation**: Quantify fault impact

### 3. Real-World Validation
- **Hardware-in-the-Loop**: Test with actual actuators
- **Flight Testing**: Validate fault scenarios in flight
- **Failure Analysis**: Compare with actual failure data

---

## Safety Considerations

### 1. Critical Faults
- **Single Point Failures**: Identify critical actuators
- **Redundancy Requirements**: Determine backup systems needed
- **Emergency Procedures**: Define safe landing protocols
- **Pilot Training**: Train for fault scenarios

### 2. Fault Tolerance Design
- **Control Redundancy**: Multiple control surfaces
- **Actuator Redundancy**: Backup actuators
- **Sensor Redundancy**: Multiple sensors per measurement
- **Power Redundancy**: Backup power systems

### 3. Certification Requirements
- **Fault Tree Analysis**: Systematic failure analysis
- **Reliability Requirements**: Meet safety standards
- **Testing Requirements**: Comprehensive fault testing
- **Documentation**: Complete fault analysis documentation

---

## Example Usage Scenarios

### 1. Fault-Tolerant Control Design
```python
# Design controller that handles actuator faults
controller = design_fault_tolerant_controller(actuator_faults)
```

### 2. Mission Planning
```python
# Plan mission considering potential faults
mission_plan = plan_robust_mission(actuator_faults)
```

### 3. Emergency Procedures
```python
# Define emergency procedures for each fault
emergency_procedures = define_emergency_procedures(actuator_faults)
```

---

## Extensions & Future Work

### 1. Advanced Fault Models
- **Intermittent Faults**: Faults that come and go
- **Progressive Faults**: Faults that worsen over time
- **Coupled Faults**: Multiple simultaneous faults
- **Environmental Faults**: Weather-related failures

### 2. Machine Learning Integration
- **Fault Prediction**: Predict faults before they occur
- **Anomaly Detection**: Detect unknown fault types
- **Adaptive Control**: Learn from fault scenarios
- **Prognostics**: Predict remaining useful life

### 3. Real-Time Fault Management
- **Online Fault Detection**: Real-time fault monitoring
- **Adaptive Control**: Automatic control reconfiguration
- **Fault Recovery**: Automatic fault recovery procedures
- **Performance Optimization**: Optimize performance under faults

---

## References

1. Patton, R. J., et al. (2013). *Fault Diagnosis and Fault-Tolerant Control Strategies for Non-Linear Systems*. Springer.
2. Isermann, R. (2011). *Fault-Diagnosis Applications: Model-Based Condition Monitoring, Actuators, Drives, Machinery, Plants, Sensors, and Fault-Tolerant Systems*. Springer.
3. Blanke, M., et al. (2016). *Diagnosis and Fault-Tolerant Control*. Springer.

---

*Document Version: 1.0*  
*Last Updated: 2025-01-27*  
*Dataset Source: Synthetic fault scenarios based on eVTOL actuator analysis*
