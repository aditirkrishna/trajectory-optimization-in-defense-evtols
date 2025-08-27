import numpy as np
import pandas as pd

# Define synthetic airfoil types
airfoils = ["NACA2412_synth", "NACA0015_synth", "CustomBlended_synth"]

# Parameters
alpha_range = np.arange(-10, 21, 1)  # AoA from -10 to +20 degrees (31 values)
Re_values = [5e5, 1e6, 2e6]          # Reynolds numbers
M_values = [0.1, 0.2, 0.3]           # Mach numbers

# Synthetic coefficients setup
Cd_min = {"NACA2412_synth": 0.02, "NACA0015_synth": 0.025, "CustomBlended_synth": 0.018}
k_drag = {"NACA2412_synth": 0.04, "NACA0015_synth": 0.05, "CustomBlended_synth": 0.035}
alpha0 = {"NACA2412_synth": -2, "NACA0015_synth": -1, "CustomBlended_synth": -1.5}  # zero-lift AoA (deg)
Cl_alpha = {"NACA2412_synth": 0.11, "NACA0015_synth": 0.105, "CustomBlended_synth": 0.115}  # per degree
Cm_const = {"NACA2412_synth": -0.05, "NACA0015_synth": -0.02, "CustomBlended_synth": -0.04}

# Dataset generation
data = []
for airfoil in airfoils:
    for Re in Re_values:
        for M in M_values:
            for alpha in alpha_range:
                # Lift coefficient using thin-airfoil-like linear slope
                Cl = Cl_alpha[airfoil] * (alpha - alpha0[airfoil])
                
                # Drag coefficient using parabolic polar
                Cd = Cd_min[airfoil] + k_drag[airfoil] * (Cl - 0.3)**2
                
                # Moment coefficient (constant synthetic assumption)
                Cm = Cm_const[airfoil]
                
                data.append([airfoil, alpha, Re, M, Cl, Cd, Cm, "synthetic_thin_airfoil_model"])

# Create DataFrame
df_airfoil = pd.DataFrame(data, columns=["airfoil","alpha_deg","Re","M","Cl","Cd","Cm","source"])

# Save CSV
df_airfoil.to_csv("project-datasets/0_raw/vehicle/datasheets/airfoil_curves.csv", index=False)

print("✅ airfoil_curves.csv generated with", len(df_airfoil), "rows")
print("\n📊 Dataset Summary:")
print(f"Total rows: {len(df_airfoil)}")
print(f"Airfoils: {len(airfoils)}")
print(f"Angle of attack range: {alpha_range[0]}° to {alpha_range[-1]}° ({len(alpha_range)} values)")
print(f"Reynolds numbers: {Re_values}")
print(f"Mach numbers: {M_values}")
print(f"Expected: {len(airfoils)} × {len(Re_values)} × {len(M_values)} × {len(alpha_range)} = {len(airfoils) * len(Re_values) * len(M_values) * len(alpha_range)}")

# Verify all combinations
print(f"\n📋 Verification:")
print(f"Unique airfoils: {df_airfoil['airfoil'].nunique()}")
print(f"Unique Re values: {df_airfoil['Re'].nunique()}")
print(f"Unique M values: {df_airfoil['M'].nunique()}")
print(f"Unique alpha values: {df_airfoil['alpha_deg'].nunique()}")

print("\n📋 Sample data:")
print(df_airfoil.head(10))
