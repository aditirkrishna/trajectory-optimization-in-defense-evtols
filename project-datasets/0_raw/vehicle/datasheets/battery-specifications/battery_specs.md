# Battery Specs — Dataset Documentation (battery_specs.md)

**Location (recommended)**  
`project-datasets/0_raw/vehicle/datasheets/battery_specs.csv`  
Document (this file): `project-datasets/5_reports/battery_specs.md`

---

## 0. Purpose (short)
This dataset captures battery performance parameters across multiple chemistries, temperatures, and discharge (C-rate) conditions for a synthetic defense eVTOL.  
It is intended for: energy budgeting, mission endurance prediction, SOC simulation, thermal/derating modeling, and use inside trajectory optimizers and flight simulators.

---

## 1. File(s) & schema

### 1.1 Baseline CSV schema (minimal sheet used earlier)
```
chemistry,temp_C,C_rate,capacity_Ah,energy_Wh,specific_energy_Whkg,round_trip_efficiency_percent,source
```

### 1.2 Recommended extended CSV schema (preferred — more explanatory & ML-ready)
```
chemistry,
temp_C,                 # ambient temperature / cell temp (°C)
C_rate,                 # discharge rate (C)
capacity_Ah,            # effective usable capacity at that condition (Ah)
nominal_cell_voltage_V, # nominal cell voltage (V) (single-cell basis)
pack_voltage_V,         # nominal pack voltage (V) used for energy calculations
energy_Wh,              # energy = capacity_Ah * pack_voltage_V (Wh)
specific_energy_Whkg,   # Wh per kg (pack-level)
internal_resistance_ohm,# internal DC resistance (Ohm) for pack or per cell (state dependent)
round_trip_efficiency_percent, # coulombic & conversion losses (%)
SoH_percent,            # State-of-Health (0-100), optional (cycle aging)
pack_mass_kg,           # mass of the battery pack (kg)
max_discharge_current_A,# = capacity_Ah * C_rate
max_continuous_power_W, # pack_voltage_V * max_discharge_current_A
notes,                  # free-form notes / derivation / source
source                  # "synthetic_multi_condition" or datasheet link
```

**Why extended schema?** downstream energy integration and ML training work best when pack_voltage, internal resistance, mass and SoH are available. If you prefer a minimal CSV for human readability keep the baseline, but add a separate `battery_specs_extended.csv` for ML.

---

## 2. Scope & ranges used in the synthetic dataset
- Chemistries (synthetic set): `Li-ion_NMC`, `Li-S`, `SolidState`  
- Temperature points: `-20, 0, 20, 40` °C (ambient/cell)  
- C-rates: `0.5, 1, 2, 5` (discharge multipliers relative to capacity)  
- Baseline capacities (cell-level or pack equivalent in our earlier generation):  
  - Li-ion_NMC: 200 Ah (base)  
  - Li-S: 300 Ah  
  - SolidState: 250 Ah  
- Nominal cell voltage (single-cell baseline used for packed-energy approximation in earlier scripts): `3.7 V` (per cell). For pack calculations we recommend using a pack voltage (e.g., 400 V) — described below.

> Note: these numbers are **synthetic**. They are chosen to produce realistic order-of-magnitude behavior for an eVTOL research platform. Replace with vendor datasheets when available.

---

## 3. Key definitions (glossary — one-stop meanings)

- **chemistry**: battery chemistry/type (identifies electrochemical family). Affects energy density, internal resistance, thermal sensitivity.
- **temp_C**: ambient or cell temperature in degrees Celsius. Battery capacity and internal resistance vary strongly with temperature.
- **C_rate**: discharge multiplier relative to capacity. `1C` means a current equal to the nominal capacity (Ah) discharged in 1 hour. `2C` is double that current (discharged in 0.5 h).
  - `I (A) = capacity_Ah * C_rate`
- **capacity_Ah**: usable amp-hours at specified condition (Ah). Capacity derates with temperature, high C-rate, and aging (SoH).
- **nominal_cell_voltage_V**: nominal single-cell voltage (e.g., 3.7 V for many Li-ion cells).
- **pack_voltage_V**: battery pack nominal voltage (series cells × cell nominal voltage). eVTOL packs often use high voltages (e.g., 300–800 V).
- **energy_Wh**: electrical energy available (Wh). `energy_Wh = capacity_Ah * pack_voltage_V`.
  - If only cell-level values are given, `energy_cell_Wh = capacity_Ah * nominal_cell_voltage_V`. Convert to pack energy by multiplying by `cells_in_series`.
- **specific_energy_Whkg**: Wh/kg — pack-level energy density. Useful for coupling battery mass to vehicle mass.
- **internal_resistance_ohm**: DC internal resistance (Ω). Causes voltage sag under load: `V_load = V_oc - I*R_internal`.
  - Resistive losses (heat): `P_loss = I^2 * R_internal`.
- **round_trip_efficiency_percent**: accounts for conversion losses, inverter/ESC losses, coulombic inefficiency; used to convert mechanical power demand to battery withdrawal.
- **SoH_percent**: State-of-Health; fraction of original capacity remaining after aging/cycles (0–100).
- **pack_mass_kg**: mass of the battery pack, derived from `energy_Wh / specific_energy_Whkg`.
- **max_discharge_current_A**: current limit defined by C_rate and capacity: `I_max = capacity_Ah * C_rate`.
- **max_continuous_power_W**: `pack_voltage_V * I_max`.
- **notes / source**: provenance of the row (synthetic formula, vendor datasheet link, lab test).

---

## 4. Physical models & equations (mathematical backbone)

Below are the standard formulas used to compute fields and how to apply derating.

### 4.1 Pack energy (Wh)
If `capacity_Ah` refers to pack-level capacity and `pack_voltage_V` is pack nominal voltage:
\[
E_\text{pack} \;[\mathrm{Wh}] = \text{capacity}_{\mathrm{Ah}} \times \text{pack\_voltage}_V
\]

If capacity is per cell and you have `N_{series}` cells:
\[
E_\text{pack} = \text{capacity}_{\mathrm{Ah}} \times (\text{nominal\_cell\_voltage}_V \times N_{series})
\]

### 4.2 Current & power
- Current at a given C-rate:
\[
I\; [\mathrm{A}] = \mathrm{capacity}_{\mathrm{Ah}} \times \text{C\_rate}
\]
- Electrical power drawn from pack:
\[
P_\text{elec}\; [\mathrm{W}] = V_\text{pack} \times I
\]
- Mechanical power required by motors \(P_{mech}\) is converted to electrical:
\[
P_\text{elec} = \frac{P_{mech}}{\eta_\text{motor} \cdot \eta_\text{ESC}} / (\eta_\text{battery\_rt})
\]
where \(\eta_\text{battery\_rt}\) is round-trip battery efficiency (accounting for conversion losses).

### 4.3 Voltage sag model (1st order)
Open-circuit voltage \(V_{oc}\) varies with SoC; simple linearization:
\[
V_{load}(t) = V_{oc}(\text{SoC}) - I(t) \cdot R_{int}
\]
- \(R_{int}\): internal resistance (Ω)
- \(I(t)\): instantaneous current draw

### 4.4 Energy consumed over time (integration)
For a time-step \(\Delta t\) seconds:
\[
\Delta Q \;[\mathrm{Ah}] = I \times \frac{\Delta t}{3600}
\]
SoC update (∞%):
\[
\text{SoC}_{t+\Delta t} = \text{SoC}_{t} - \frac{\Delta Q}{\text{capacity\_Ah}} \times 100
\]

Energy consumed (Wh) during \(\Delta t\):
\[
\Delta E = P_\text{elec} \times \frac{\Delta t}{3600}
\]

### 4.5 Temperature and C-rate derating (synthetic model used)
A simple, explainable derating function (used in the earlier generator):
\[
\text{cap\_factor}(T,C) = \max\!\left(0.5,\; \left(1 - \frac{|T-20|}{100}\right)\cdot (1 - 0.05\cdot (C-1)) \right)
\]
\[
\text{capacity\_Ah} = \text{capacity\_base} \times \text{cap\_factor}
\]
This is **synthetic and conservative**: it reduces capacity gently with low/high temperature and more strongly with high C-rate.

**Alternative physics-based model** (optional; recommended for realism):
- Use Arrhenius relationship for rate-limited processes, or manufacturer-provided derating curves:
\[
\text{capacity}(T) \approx \text{capacity}_{20^\circ C} \cdot \left(1 - a \cdot (20 - T)\right) \quad \text{for } T < 20^\circ C
\]
with coefficient \(a\) determined experimentally.

---

## 5. How the synthetic dataset was generated (algorithm summary)

1. For each `chemistry` (Li-ion_NMC, Li-S, SolidState) define:
   - `base_capacity_Ah`, `base_specific_energy_Whkg`, `nominal_cell_voltage_V`.
2. For each `temp_C` and `C_rate`:
   - Compute `cap_factor` (see §4.5).
   - `capacity_Ah = base_capacity_Ah * cap_factor`.
   - `energy_Wh = capacity_Ah * nominal_cell_voltage_V` (earlier script used 3.7 V; for pack-level use pack_voltage_V).
   - `specific_energy_Whkg = base_specific_energy_Whkg * cap_factor` (approx).
   - `round_trip_efficiency_percent` = base_efficiency - small penalties for temp & high C-rate.
3. Estimate `pack_mass_kg = energy_Wh / specific_energy_Whkg`.
4. Compute `internal_resistance_ohm` heuristically as a function of chemistry and temperature (lower at warm temps).

**Note:** this generator is intentionally simple and deterministic so the dataset is reproducible. When you receive vendor datasheets, replace the synthetic derivation with measured curves.

---

## 6. Example rows (sample CSV snippet)
```csv
chemistry,temp_C,C_rate,capacity_Ah,nominal_cell_voltage_V,pack_voltage_V,energy_Wh,specific_energy_Whkg,internal_resistance_ohm,round_trip_efficiency_percent,SoH_percent,pack_mass_kg,max_discharge_current_A,max_continuous_power_W,notes,source
Li-ion_NMC,20,1.0,200,3.7,400,80000,250,0.020,95,100,320,200,80000,"base case synthetic pack (pack_voltage=400V)",synthetic_multi_condition
Li-ion_NMC,-20,2.0,150,3.7,400,60000,230,0.035,88,95,260,300,120000,"cold derated and higher internal resistance",synthetic_multi_condition
Li-S,20,1.0,300,3.7,400,120000,400,0.015,90,100,300,300,120000,"higher energy density (synthetic)",synthetic_multi_condition
SolidState,40,0.5,220,3.7,400,88000,350,0.018,92,100,251.4,110,44000,"hot but less derating",synthetic_multi_condition
```

---

## 7. Example: Using the data in a simulation step (worked numeric example)

**Given**

* Row: Li-ion_NMC, temp 20°C, C_rate 1, capacity_Ah = 200 Ah, pack_voltage_V = 400 V, internal_resistance = 0.02 Ω, round_trip_eff=95%.
* Motor/mechanical demand at time t: `P_mech = 50 kW`.
* Motor + ESC combined efficiency = `η_motor·η_ESC = 0.92` (92%).

**Compute**

1. Electrical power before battery losses:

   $$
   P_{elec\_inverter} = \frac{P_{mech}}{0.92} = \frac{50000}{0.92} \approx 54347 \,\mathrm{W}
   $$
2. Battery-side power accounting round-trip inefficiency (approx):

   $$
   P_{pack\_draw} = \frac{P_{elec\_inverter}}{0.95} \approx 57207 \,\mathrm{W}
   $$
3. Current drawn:

   $$
   I = \frac{P_{pack\_draw}}{V_{pack}} = \frac{57207}{400} \approx 143.0 \,\mathrm{A}
   $$
4. Check within `max_discharge_current_A` (200 A for 200 Ah × 1C) — OK.
5. Voltage sag:

   $$
   V_{load} = V_{oc} - I R_{int} \approx 400 - 143\times 0.02 = 400 - 2.86 = 397.14\,\mathrm{V}
   $$
6. Energy use for 1 minute (Δt = 60 s):

   $$
   \Delta E = P_{pack\_draw} \cdot \frac{60}{3600} = 57207 \times 0.0166667 \approx 953.45\,\mathrm{Wh}
   $$
7. SoC drop:

   $$
   \Delta Q [\mathrm{Ah}] = \frac{I \cdot 60}{3600} = \frac{143\cdot 60}{3600} \approx 2.383\,\mathrm{Ah}
   $$

   Relative SoC drop: $2.383 / 200 \times 100 = 1.19\% $ SoC for that minute.

This is the sort of per-step update your SITL/MPC loop needs — use `internal_resistance_ohm` to model sag and `round_trip_efficiency` to link mechanical to electrical.

---

## 8. Preprocessing & recommended transformations for ML

* **Normalize continuous features** (temp, C_rate, capacity_Ah, pack_voltage_V) using StandardScaler saved with dataset metadata.
* **One-hot encode `chemistry`** or embed with small learned embeddings.
* **Log-transform internal_resistance_ohm** if range spans orders of magnitude.
* **Create derived scalar features**:

  * `max_discharge_current_A = capacity_Ah * C_rate`
  * `pack_energy_kWh = energy_Wh / 1000`
  * `energy_density_Wh_per_kg = specific_energy_Whkg`
* **Aggregate** multiple rows into lookup-table structures or interpolate between rows for continuous param sweeps.

---

## 9. Validation & QA tests

Implement automated checks when ingesting CSV rows:

* **Unit checks**:

  * `capacity_Ah > 0`
  * `pack_voltage_V > 0`
  * `energy_Wh ≈ capacity_Ah * pack_voltage_V` (allow small error tolerance)
  * `specific_energy_Whkg > 0`
* **Range checks**:

  * `round_trip_efficiency_percent` in `[40, 100]`
  * `internal_resistance_ohm` > 0
* **Plausibility**:

  * `pack_mass_kg ≈ energy_Wh / specific_energy_Whkg` (tolerance 10–20%)
* **Monotonicity** (optional):

  * For fixed chemistry & temperature, capacity should not increase with higher C_rate.
* **File-level**:

  * Check no missing required columns.
  * Sanity-check that `source` field is present.

Implement these checks in `scripts/validate_layers.py` or a new `scripts/validate_battery.py`.

---

## 10. How to extend / recommended richer studies

To be ML-grade and robust for defense scenarios, extend dataset with:

* **Finer C-rate grid**: 0.1 → 6.0 in steps of 0.1 (gives many rows).
* **More temperature points**: -40 → +60 °C in 5–10 °C steps.
* **State-of-Health (SoH)**: simulate aging with cycle counts (0–100k cycles).
* **Transient response**: time-dependent internal resistance and thermal coupling (R(t), T(t)), thermal runaway limits.
* **Pack configuration variants**: pack_voltage 300V, 400V, 600V; series/parallel counts.
* **Cell-to-cell variance**: Monte-Carlo perturbations to model pack imbalance.
* **Empirical fits**: if you later measure real battery test data, add `tolerance` columns and curve-fit parameters.

---

## 11. Integration tips — connecting battery dataset to other datasets

* **Rotors → mechanical power**: For each rotor/time-step compute mechanical power from rotor dataset (torque × ω). Sum rotor mechanical power → apply drivetrain efficiencies → battery draw (use `round_trip_efficiency_percent`).
* **Mass coupling**: `pack_mass_kg` feeds back into `mass_inertia.csv` (update mass and CG). Keep a `vehicle_config.json` that references which battery row was used.
* **Thermal coupling**: If ambient `temp_C` is high and battery internal resistance increases, motor/ESC cooling may be affected — include margin.

---

## 12. Example generator pseudo-code (quick overview)

This is the logic used to create the synthetic `battery_specs.csv`:

```
for chem in chemistries:
  base_capacity = chem_base_capacity[chem]
  base_energy_density = chem_base_energy_density[chem]
  for temp in temps:
    for C in C_rates:
      cap_factor = max(0.5, (1 - abs(temp-20)/100) * (1 - 0.05*(C-1)))
      capacity_Ah = base_capacity * cap_factor
      nominal_cell_voltage_V = 3.7
      pack_voltage_V = 400  # default; user should set
      energy_Wh = capacity_Ah * pack_voltage_V
      specific_energy_Whkg = base_energy_density * cap_factor
      pack_mass_kg = energy_Wh / specific_energy_Whkg
      internal_resistance_ohm = base_R[chem] * (1 + 0.01*(20-temp)) # synthetic
      round_trip_eff = base_eff[chem] - 0.3*abs(temp-20) - 2*(C-1)
      write_row(...)
```

**Important**: Replace `pack_voltage_V` with pack-specific real values when available.

---

## 13. Versioning, metadata & provenance

* Each CSV row must include `source` (e.g., `synthetic_multi_condition_v1`) and a dataset-level `README` must contain:

  * generator script hash/commit,
  * random seed (if any),
  * generation date,
  * known limitations.
* Keep raw generator script under `scripts/` (e.g., `scripts/generate_battery_specs.py`) and add a unit test that checks `energy_Wh` consistency.

---

## 15. Minimal actionable checklist for the team

1. Copy the recommended **extended CSV schema** to `battery_specs_extended.csv` and generate synthetic rows using the pseudo-code above (or the script provided earlier).
2. Save the generator script in `scripts/generate_battery_specs.py` and the dataset to `0_raw/vehicle/datasheets/`.
3. Add validation tests (`scripts/validate_battery.py`) that run on CI.
4. Document the generator version and commit SHA in `5_reports/battery_specs.md` (this doc).
5. Link the battery row used in any SITL campaign to the `vehicle_config.json` so experiments are reproducible.

---

## 16. References & further reading (internal notes)

* When you have vendor datasheets replace synthetic values with:

  * `capacity_Ah` vs temp & C-rate tables,
  * `internal_resistance` vs SoC & temp,
  * manufacturer `pack_voltage` and `cell_count`.
* For physics-rich simulation consider implementing Thevenin battery equivalent circuit (Rint + RC branch) and thermal model.

