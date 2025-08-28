# Flight & Structural Limits — Dataset Documentation (limits.md)

**Dataset location:**  
- Raw CSV: `project-datasets/0_raw/vehicle/datasheets/limits/limits.csv`  
- Documentation (this file): `project-datasets/5_reports/limits.md`

---

## 0. Purpose
This dataset defines the **operational envelopes, performance constraints, and structural limits** of the synthetic defense eVTOL platform.  
These values constrain:
- **Trajectory optimizers** (paths must remain within legal/feasible envelopes).  
- **Simulation environments** (to reject infeasible maneuvers).  
- **Energy models** (endurance affected by climb rate, speed, payload, etc).  

---

## 1. File & Schema

### 1.1 Minimal CSV schema (baseline form we generated earlier)
```csv
payload_kg,max_takeoff_weight_kg,service_ceiling_m,max_speed_mps,max_climb_rate_mps,max_descent_rate_mps,max_bank_angle_deg,max_load_factor_g,max_rotor_rpm,min_rotor_rpm,notes,source
```

### 1.2 Extended schema (recommended, more useful for ML/optimizers)

```csv
payload_kg,                   # Payload carried (kg)
max_takeoff_weight_kg,        # Maximum takeoff mass (kg)
service_ceiling_m,            # Maximum operational altitude (m, ASL)
max_forward_speed_mps,        # Maximum horizontal flight speed (m/s)
stall_speed_mps,              # Minimum forward flight speed (m/s)
max_climb_rate_mps,           # Maximum climb rate (m/s)
max_descent_rate_mps,         # Maximum descent rate (m/s)
max_bank_angle_deg,           # Structural bank angle limit (deg)
max_pitch_angle_deg,          # Nose-up/nose-down pitch limit (deg)
max_load_factor_g,            # Positive load factor (g)
min_load_factor_g,            # Negative load factor (g) for inverted/steep maneuvers
max_rotor_rpm,                # Upper rotor rotational speed limit (RPM)
min_rotor_rpm,                # Idle or autorotation lower limit (RPM)
gust_limit_mps,               # Max tolerable gust before loss of control (m/s)
structural_margin_factor,     # Synthetic safety factor for defense ops
notes,                        # Explanation of assumption/derivation
source                        # synthetic or datasheet/prototype link
```

---

## 2. Glossary of Variables

* **payload\_kg** — payload mass carried by the aircraft. Affects performance and endurance.
* **max\_takeoff\_weight\_kg (MTOW)** — maximum allowed gross takeoff weight (vehicle + payload + fuel/battery).
* **service\_ceiling\_m** — maximum altitude at which aircraft can sustain a minimum climb rate (often \~0.5 m/s).
* **max\_forward\_speed\_mps** — maximum horizontal flight speed limited by rotor power, aerodynamic drag, or compressibility.
* **stall\_speed\_mps** — minimum forward flight speed required for winged eVTOL configurations; for multicopters, this is near 0 but often defined by control authority margins.
* **max\_climb\_rate\_mps** — vertical climb rate limit (m/s). Determined by thrust-to-weight ratio and thermal limits.
* **max\_descent\_rate\_mps** — safe descent rate limit. Exceeding this can cause vortex ring state (VRS) in multirotors.
* **max\_bank\_angle\_deg** — roll angle limit (deg). Bank angle relates to lateral acceleration and load factor.
* **max\_pitch\_angle\_deg** — pitch angle limit (deg). Prevents loss of control in aggressive nose-up/down maneuvers.
* **max\_load\_factor\_g** — maximum positive g-load the structure can sustain without failure.
* **min\_load\_factor\_g** — maximum negative g-load (push-over maneuvers).
* **max\_rotor\_rpm / min\_rotor\_rpm** — operational range of rotor revolutions per minute. Min = autorotation idle; Max = mechanical/structural stress limit.
* **gust\_limit\_mps** — maximum gust speed that can be tolerated structurally. Useful for environment × vehicle coupling.
* **structural\_margin\_factor** — synthetic margin of safety multiplier (defense design typically uses >1.5).

---

## 3. Mathematical Background

### 3.1 Load factor vs bank angle

Load factor $n$ in a coordinated level turn is:

$$
n = \frac{1}{\cos(\phi)}
$$

where $\phi$ = bank angle (deg).
For example:

* At $60^\circ$ bank, $n = 2.0g$.
* At $75^\circ$ bank, $n = 3.86g$.

Thus, **max bank angle** and **max load factor** are tightly coupled.

---

### 3.2 Climb and descent rates

* Maximum climb rate:

$$
V_{climb,max} = \frac{(T_{avail} - W)}{W} \cdot V
$$

where $T_{avail}$ = available thrust, $W$ = weight, $V$ = velocity component.
Simplified in multicopters to thrust-to-weight ratio:

$$
V_{climb,max} \approx \frac{(T/W - 1) \cdot g}{\text{mass}} \quad \text{(synthetic)}
$$

* Maximum descent rate limited by avoiding vortex ring state:

$$
V_{descent,max} \approx 0.7 \sqrt{\frac{T}{2 \rho A}}
$$

where $A$ = rotor disk area, $\rho$ = air density.

---

### 3.3 Structural margins

Defense airframes often require a **structural safety factor ≥ 1.5**:

$$
n_{limit} = n_{ultimate} / \text{margin}
$$

---

### 3.4 Rotor RPM ranges

* Max RPM is set by blade tip Mach number:

$$
M_{tip} = \frac{\pi D \cdot RPM}{60 a}
$$

where $a$ = speed of sound (\~343 m/s at sea level).
Keep $M_{tip} < 0.75$ to avoid compressibility issues.

* Min RPM must be high enough to avoid loss of lift/control.

---

## 4. Synthetic Assumptions Used

* **Payload range:** 0 – 600 kg (defense payload spectrum).
* **MTOW:** 1800–2500 kg depending on payload.
* **Service ceiling:** 4000–6000 m (defense multirotor hybrid).
* **Max forward speed:** 70–120 m/s (≈250–430 km/h).
* **Max climb:** 6–15 m/s.
* **Max descent:** 4–8 m/s (derated for VRS risk).
* **Bank angle:** up to 65° (≈2.4 g).
* **Load factors:** +3.0 g, -1.0 g (synthetic conservative).
* **Rotor RPM:** 1500–5000 (depends on rotor size).
* **Gust tolerance:** 20–30 m/s.
* **Structural margin factor:** 1.5.

---

## 5. Example CSV snippet

```csv
payload_kg,max_takeoff_weight_kg,service_ceiling_m,max_forward_speed_mps,stall_speed_mps,max_climb_rate_mps,max_descent_rate_mps,max_bank_angle_deg,max_pitch_angle_deg,max_load_factor_g,min_load_factor_g,max_rotor_rpm,min_rotor_rpm,gust_limit_mps,structural_margin_factor,notes,source
0,2000,5000,120,0,12,6,65,30,3.0,-1.0,4800,1500,25,1.5,"baseline empty payload","synthetic"
300,2200,4800,115,0,10,6,60,25,2.8,-0.8,4700,1500,25,1.5,"medium payload, derated climb","synthetic"
600,2500,4500,105,0,8,5,55,20,2.5,-0.8,4600,1500,20,1.5,"heavy payload, reduced ceiling","synthetic"
```

---

## 6. Usage in Optimizers & Simulators

* **Trajectory optimizers**:

  * Must keep altitude ≤ service ceiling.
  * Must keep velocity between stall and max speed.
  * Must reject maneuvers requiring bank angle > limit or g-load > limit.
  * Must respect climb/descent bounds for feasible vertical transitions.

* **SITL (simulation-in-the-loop)**:

  * Rotor dynamics limited between min/max RPM.
  * Gusts injected into environment must not exceed gust\_limit\_mps.
  * Overstress triggers structural failure mode.

---

## 7. Validation checks

Automated checks for the CSV:

* `max_takeoff_weight_kg >= payload_kg + empty_mass`
* `stall_speed_mps >= 0`
* `max_climb_rate_mps > 0`, `max_descent_rate_mps > 0`
* `max_load_factor_g >= 1`
* `max_bank_angle_deg` consistent with `max_load_factor_g` via formula (§3.1)
* `max_rotor_rpm > min_rotor_rpm`

---

## 8. Extension ideas

* Make rows **altitude-dependent** (e.g., reduce max climb at 4000 m).
* Add **thermal derating** (high temp reduces max continuous power).
* Add **dynamic gust response curves** (gust tolerance vs altitude).
* Model **rotor tip Mach constraint** directly instead of static RPM limit.

---

## 9. Dataset Statistics

### Coverage Matrix
- **Payload Levels**: 0 kg, 300 kg, 600 kg (defense spectrum)
- **Performance Degradation**: Linear reduction with payload
- **Safety Margins**: Conservative for defense operations
- **Environmental Factors**: Gust tolerance and structural margins

### Performance Ranges
| Payload (kg) | Max Speed (m/s) | Max Climb (m/s) | Service Ceiling (m) | Max Load Factor (g) |
|--------------|-----------------|-----------------|-------------------|-------------------|
| 0 | 120 | 12 | 5000 | 3.0 |
| 300 | 115 | 10 | 4800 | 2.8 |
| 600 | 105 | 8 | 4500 | 2.5 |

### Structural Considerations
- **Bank Angle Limits**: 55°-65° (payload dependent)
- **Pitch Angle Limits**: 20°-30° (payload dependent)
- **Rotor RPM Range**: 1500-4800 RPM (payload dependent)
- **Gust Tolerance**: 20-25 m/s (payload dependent)

---

## 10. Related Datasets
- `mass_inertia.csv`: Vehicle mass properties for dynamics
- `rotor_thrust_curves_multi.csv`: Propulsive characteristics
- `battery_specs.csv`: Energy storage and endurance limits
- `airfoil_curves.csv`: Aerodynamic performance data

---

## 11. References
- Aircraft performance theory and flight dynamics
- eVTOL structural design guidelines
- Defense vehicle operational requirements
- Rotorcraft aerodynamics and control limits
