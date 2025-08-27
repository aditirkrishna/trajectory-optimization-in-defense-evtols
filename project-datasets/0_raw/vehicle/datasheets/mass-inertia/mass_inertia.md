# Mass & Inertia Dataset Documentation

## Purpose
Defines mass properties and inertial parameters of defense eVTOL for 6-DOF flight dynamics, stability analysis, and trajectory simulation.

## Dataset Schema
```
mass_kg,cg_x_m,cg_y_m,cg_z_m,Ixx,Iyy,Izz,source
```

## Specifications

| Parameter | Value | Unit | Notes |
|-----------|-------|------|-------|
| mass_kg   | 1500.0 | kg | Total takeoff mass |
| cg_x_m    | 0.50  | m  | CG longitudinal (forward offset) |
| cg_y_m    | 0.00  | m  | CG lateral (symmetry) |
| cg_z_m    | -0.20 | m  | CG vertical (below rotor plane) |
| Ixx       | 8781.25 | kg·m² | Roll moment of inertia |
| Iyy       | 12500.00 | kg·m² | Pitch moment of inertia |
| Izz       | 5281.25 | kg·m² | Yaw moment of inertia |
| source    | synthetic_prism_model | Generation method |

## Assumptions
1. **Geometry**: Rectangular prism (8m×6m×2.5m)
2. **Mass Distribution**: Uniform density
3. **CG Placement**: Forward and below rotor plane for stability

## Mathematical Derivations

Rectangular prism formulas:
- **Roll (x-axis)**: Ixx = (1/12) × m × (h² + w²)
- **Pitch (y-axis)**: Iyy = (1/12) × m × (l² + h²)  
- **Yaw (z-axis)**: Izz = (1/12) × m × (l² + w²)

Where: m=1500kg, l=8m, w=6m, h=2.5m

## Variable Glossary

| Variable | Meaning | Unit | eVTOL Range |
|----------|---------|------|-------------|
| mass_kg | Total aircraft mass | kg | 1000–2500 |
| cg_x_m | Longitudinal CG (fwd +) | m | -1.0 → +1.0 |
| cg_y_m | Lateral CG (right +) | m | -0.2 → +0.2 |
| cg_z_m | Vertical CG (down +) | m | -0.5 → +0.5 |
| Ixx | Roll inertia | kg·m² | 5000–10000 |
| Iyy | Pitch inertia | kg·m² | 8000–15000 |
| Izz | Yaw inertia | kg·m² | 3000–7000 |

## Usage Notes
- Input for 6-DOF flight dynamics solvers
- Scale with payload mass changes
- Replace with CAD/flight-test data when available

## Coordinate System
- **X-axis**: Forward along fuselage
- **Y-axis**: Right (starboard)  
- **Z-axis**: Downward
- **Origin**: Aircraft geometric center

## Validation
- ✅ CG within realistic bounds
- ✅ Iyy > Ixx > Izz (typical aircraft ratios)
- ✅ Consistent with eVTOL geometry
