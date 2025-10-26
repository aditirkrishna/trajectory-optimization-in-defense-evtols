# Mass & Inertia Dataset Documentation (mass_inertia.md)

**Dataset location:**  
- Raw CSV: `project-datasets/0_raw/vehicle/datasheets/mass_inertia.csv`  
- Documentation (this file): `project-datasets/5_reports/mass_inertia.md`

---

## 0. Purpose
This dataset defines the **mass properties and inertial parameters** of the defense eVTOL reference configuration.  
These are fundamental inputs for:
- **Equations of motion (6-DOF)** in flight dynamics,  
- **Stability & control analysis** (longitudinal, lateral),  
- **Trajectory optimization** (CG placement affects trim),  
- **Energy modeling** (inertia affects maneuver cost).  

---

## 1. File & Schema

### 1.1 Current CSV schema
```csv
mass_kg,cg_x_m,cg_y_m,cg_z_m,Ixx,Iyy,Izz,source
```

### 1.2 Extended schema (recommended)

```csv
mass_kg,                 # Total takeoff mass (kg)
empty_mass_kg,           # Vehicle empty mass (kg)
payload_mass_kg,         # Payload component (kg)
battery_mass_kg,         # Battery component (kg)
cg_x_m,                  # Longitudinal CG location (m, fwd+)
cg_y_m,                  # Lateral CG location (m, right+)
cg_z_m,                  # Vertical CG location (m, down+)
Ixx_kgm2,                # Moment of inertia about x-axis (roll)
Iyy_kgm2,                # Moment of inertia about y-axis (pitch)
Izz_kgm2,                # Moment of inertia about z-axis (yaw)
Ixy_kgm2,                # Product of inertia (optional, coupling)
Ixz_kgm2,                # Product of inertia (optional, coupling)
Iyz_kgm2,                # Product of inertia (optional, coupling)
geometry_assumption,      # Simplified geometry basis (prism, blended wing, etc.)
notes,                   # Description of synthetic assumption
source                   # Origin (synthetic, CAD, vendor datasheet)
```

---

## 2. Glossary of Variables

* **mass\_kg** — total mass of aircraft (kg). Includes empty weight, payload, and battery.
* **empty\_mass\_kg** — structural + avionics weight without payload/energy.
* **payload\_mass\_kg** — removable payload (sensors, cargo, defense package).
* **battery\_mass\_kg** — mass of the energy storage system (Li-ion pack).
* **cg\_x\_m, cg\_y\_m, cg\_z\_m** — center of gravity coordinates in body frame (origin at rotor hub plane).
* **Ixx\_kgm2** — roll inertia, resists rolling motion.
* **Iyy\_kgm2** — pitch inertia, resists nose-up/down rotation.
* **Izz\_kgm2** — yaw inertia, resists rotation about vertical axis.
* **Ixy, Ixz, Iyz** — products of inertia (for asymmetry; set to zero here since symmetry assumed).
* **geometry\_assumption** — indicates which simplified geometry was used (rectangular prism here).
* **notes** — explanation of how values were derived.
* **source** — dataset origin ("synthetic\_prism\_model" in this case).

---

## 3. Mathematical Background

### 3.1 Mass Properties

Total mass:

$$
m_{total} = m_{empty} + m_{payload} + m_{battery}
$$

CG (vector):

$$
\mathbf{r_{CG}} = \frac{\sum m_i \mathbf{r_i}}{\sum m_i}
$$

---

### 3.2 Moments of Inertia (rectangular prism assumption)

For a prism with mass $m$, length $l$, width $w$, height $h$:

* Roll axis (x):

$$
I_{xx} = \frac{1}{12} m (h^2 + w^2)
$$

* Pitch axis (y):

$$
I_{yy} = \frac{1}{12} m (l^2 + h^2)
$$

* Yaw axis (z):

$$
I_{zz} = \frac{1}{12} m (l^2 + w^2)
$$

Products of inertia (assuming symmetry):

$$
I_{xy} = I_{xz} = I_{yz} = 0
$$

---

### 3.3 Synthetic Geometry Used

* **Length (l):** 8.0 m
* **Width (w):** 6.0 m
* **Height (h):** 2.5 m
* **Mass (m):** 1500 kg (synthetic baseline)

Results:

* $Ixx = 8781.25 \, \text{kg·m}^2$
* $Iyy = 12500.0 \, \text{kg·m}^2$
* $Izz = 5281.25 \, \text{kg·m}^2$

---

## 4. Synthetic Assumptions Used

1. Mass distribution is uniform.
2. CG slightly forward (x=+0.5 m) and below rotor plane (z=-0.2 m) for stability.
3. Symmetry across y-axis, so cg\_y\_m = 0.
4. Product of inertia terms neglected (assumed zero).
5. Geometry approximated as a prism instead of blended wing/cylindrical body.

---

## 5. Example CSV snippet

```csv
mass_kg,empty_mass_kg,payload_mass_kg,battery_mass_kg,cg_x_m,cg_y_m,cg_z_m,Ixx_kgm2,Iyy_kgm2,Izz_kgm2,Ixy_kgm2,Ixz_kgm2,Iyz_kgm2,geometry_assumption,notes,source
1500,900,300,300,0.5,0.0,-0.2,8781.25,12500.0,5281.25,0,0,0,"rectangular_prism","Uniform mass distribution; CG adjusted forward & below rotor plane","synthetic_prism_model"
```

---

## 6. Usage in Optimizers & Simulators

* **Trajectory optimizers**:

  * CG location feeds into trim conditions.
  * Inertia affects dynamic constraints during maneuvering.

* **SITL (Simulation-in-the-loop)**:

  * Mass & inertia matrix is loaded into physics engine.
  * Rotational acceleration = Torque / Inertia.
  * Non-zero products of inertia can later be used to model asymmetric payloads.

* **Energy models**:

  * Higher inertia → higher energy cost in aggressive maneuvers.

---

## 7. Validation Checks

* `mass_kg = empty_mass_kg + payload_mass_kg + battery_mass_kg`
* CG values should remain within ±1.0 m of body centerline (realistic constraint).
* Inertia values must be positive and consistent with geometry.
* If products of inertia ≠ 0, they must satisfy inertia tensor symmetry.

---

## 8. Extension Ideas

* Replace prism assumption with **blended-wing inertia** from CAD model.
* Add **payload shift scenarios** (off-axis CG loads).
* Model **fuel burn / battery depletion** (time-varying CG).
* Provide **full inertia tensor** (not just principal axes).

---

### End of `mass_inertia.md`
