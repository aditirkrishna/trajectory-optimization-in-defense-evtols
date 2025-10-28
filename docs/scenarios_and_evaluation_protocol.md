# Scenarios and Evaluation Protocol

This document defines the standardized scenario suite and the evaluation protocol used to assess the eVTOL trajectory optimization system across operations, environment, threats, vehicle/control, robustness, and real-time constraints. It is designed to be paper-ready and reproducible.

## 1. Scenario Taxonomy

- **Urban operations**: Dense buildings, low-altitude corridors, mixed airspace rules.
- **Long-range endurance**: Reserve margins, loiter segments, thermal derating.
- **Multi-stop logistics**: Time windows, variable payloads, depot constraints.
- **Emergency diversion**: Dynamic rerouting to safe landing sites under faults/weather.
- **GPS-denied operations**: GNSS degradation, inertial drift, map-matching fallback.
- **High-threat penetration**: Radar/EW overlays, patrol schedules, terrain masking.
- **Mountainous terrain**: Valley/ridge crossings, wind channeling, clearance.
- **Adverse weather**: Layered wind, gusts/turbulence, temperature stratification.

## 2. Canonical Scenario Definitions

Each scenario should be instantiated with a seed set S = {0,1,2,3,4} for reproducibility.

- **Urban-Short**
  - Distance: 8–15 km; Altitude: 120 m AGL nominal; Min-clearance ≥ 30 m
  - Building density ≥ 300/km²; No-fly corridors (ROZ) optional

- **Urban-Complex**
  - Distance: 12–25 km; Altitude: 100–180 m AGL; ROZ corridors; dynamic pop-up no-fly

- **Mountain-Pass**
  - Distance: 15–30 km; Altitude band: 150–600 m AGL; ≥ 2 ridge crossings

- **Long-Range**
  - Distance: 35–60 km; Cruise ≈ 35 m/s; Reserve ≥ 20% SOC; 5–10 min loiter segment

- **GPS-Denied**
  - GNSS dropout segments: 30–90 s; INS drift σ = 0.02–0.05 m/√s

- **High-Threat**
  - Distance: 12–25 km; Radar SNR threshold sweeps; 1–2 EW zones; patrol period 5–10 min

- **Multi-Drop**
  - 3–5 delivery points; service time 2–8 min; payload mass changes ±15%

- **Emergency**
  - En-route abort to nearest feasible LZ within 3–5 km; ≤ 2 min to commit

Common assumptions across scenarios:
- Wind layers at 0/100/200/300 m AGL (RBF-interpolated); turbulence via Dryden.
- Flight constraints: min-turn-radius Rmin; max climb/descent rates; min-clearance above terrain/obstacles.
- Vehicle envelope consistent with project `vehicle_layer`; thermal derating if Tbat > threshold.

## 3. Uncertainty & Robustness Models

- **Wind**: Multiplicative bias b ~ N(1, σ²), σ ∈ {0.05, 0.1, 0.2}; gusts via Dryden, length scale L ∈ {100, 300} m.
- **Threat detection**: Radar SNR threshold ±3 dB; patrol phase uniformly randomized over period.
- **Battery**: Capacity C ~ N(C0, (0.05C0)²); internal resistance +10% step if T > 40°C.
- **GPS-denied**: Position drift as random walk with spectral density q; bias resets on reacquisition.

Robust planning settings:
- Chance constraints: P(violation) ≤ α, α ∈ {0.01, 0.05, 0.1}.
- Risk measures: VaRα and CVaRα for route risk/energy.

## 4. Baselines and Ablations

Baselines for comparison:
- Straight-line (geodesic) at fixed altitude.
- A* (distance-only), A* (energy-only), A* (risk-only).
- Grid-Dijkstra with scalarized multi-criteria (commercial-like heuristic if available).

Ablations:
- Heuristics: Haversine vs 3D great-circle vs zero-heuristic.
- Grid resolution: coarse/medium/fine (e.g., 250 m / 100 m / 50 m).
- Cost weights: nominal vs distance-heavy vs risk-heavy.
- Smoothing: on vs off; curvature-bounded vs unconstrained.
- Robustness: deterministic vs chance-constrained vs CVaR.

## 5. Metrics (Optimize, Report, Compare)

Route Quality:
- Total distance (km); Flight time (s); Energy (kWh); Avg/peak power (kW).
- Risk exposure: ∫ risk(x) ds (unitless or normalized 0–1).
- Feasibility: count & severity of constraint violations (clearance, curvature, climb).

Robustness:
- Success rate under N Monte Carlo trials.
- Violation probability (≤ α target); VaR/CVaR for energy and risk.

Tracking/Control (subset, sim-in-the-loop):
- RMS tracking error (m), max deviation (m), control saturation events (#).

Performance:
- Planning time (ms); Replanning latency (ms) under map/threat update.
- Memory (MB); Expanded nodes (#); Alternatives generated (k).

Reporting:
- Mean ± std over K seeds; for robustness add VaR/CVaR and %success.

## 6. Experimental Protocol

Data & Seeds:
- Terrain/obstacles from processed layers; threats/weather from `outputs/environmenatl-intelligence/*`.
- Fixed seeds S = {0..4} for reproducibility.

Per Scenario:
- Run each baseline and the proposed method with K=5 seeds.
- If robust, perform N=200 Monte Carlo samples per seed to estimate violation probability and CVaR.
- For multi-objective, compute Pareto fronts; report knee-point for headline comparisons and include full fronts in figures.

Online Replanning:
- Insert a pop-up constraint (e.g., new ROZ) at t = 30% route progress; log replanning latency and quality degradation.

Controller Coupling (subset):
- Track smoothed trajectory; report RMS error and saturation events.

Statistical Testing:
- Wilcoxon signed-rank test between proposed vs strongest baseline for primary metrics (distance, energy, risk, time).
- Report effect sizes (Cliff’s delta) and p-values with Holm–Bonferroni correction.

## 7. Tables and Figures (Recommended)

- **Table 1 (Scenario Suite)**: name, distance, altitude band, constraints, threat/weather toggles.
- **Table 2 (Main Results)**: Distance, Energy, Risk, Time, Violations, Planning time (mean ± std) by method.
- **Table 3 (Robustness)**: Success rate, violation probability, VaR/CVaR (energy, risk) by scenario.
- **Figure 1**: Pareto fronts (distance–energy–risk) with knee-point.
- **Figure 2**: Replanning latency vs quality degradation after pop-ups.
- **Figure 3**: Tracking performance (RMS error, saturation) on representative trajectories.
- **Figure 4**: Qualitative urban overlays: route + clearance + risk heatmap.

## 8. Reproducibility & Artifacts

- **Configs**: YAML per scenario (start/goal, altitude band, toggles).
- **Seeds**: Set and documented for each run.
- **Logs**: Per-run JSON metrics; aggregation script to produce tables.
- **Artifacts**: Routes (.json/.csv), plots (.png/.pdf), benchmark summaries (.json/.txt).

Directory pointers:
- Planning: `outputs/mission-results/*`
- Environment: `outputs/environmenatl-intelligence/*`
- Vehicle/control: `outputs/vehicle-simulation/*`
- Benchmarks: `outputs/benchmarking/*`

## 9. Claims and Acceptance Criteria

Defensible Claims (examples):
- Across urban/mountainous/high-threat scenarios, the proposed method:
  - Achieves lower CVaR risk at comparable energy/time, or
  - Reduces energy by X% at matched risk/time, or
  - Reduces replanning latency by Y% under pop-up constraints.
- Robust planner satisfies P(violation) ≤ α with empirical confidence bounds.
- Controller tracks smoothed trajectories with ≤ Z m RMS under nominal winds.

Acceptance Criteria:
- Statistically significant improvements (Wilcoxon p < 0.05 after correction) on ≥ 2 primary metrics in ≥ 5/8 scenarios.
- Robustness targets met for α ∈ {0.05, 0.1} in robust configurations.

## 10. Ready-to-Fill Scenario Cards (Template)

| Scenario | Start (lat,lon,alt) | Goal (lat,lon,alt) | Distance (km) | Alt band (m AGL) | Constraints | Threats/Weather | Notes |
|---|---|---|---:|---:|---|---|---|
| Urban-Short |  |  |  | 120 | Min-clear ≥ 30 m | Buildings dense |  |
| Urban-Complex |  |  |  | 100–180 | ROZ corridors; pop-ups |  |  |
| Mountain-Pass |  |  |  | 150–600 | Ridge crossings ≥ 2 | Wind channeling |  |
| Long-Range |  |  |  | Cruise 120–200 AGL | Reserve ≥ 20% SOC | Loiter segment |  |
| GPS-Denied |  |  |  |  | GNSS drop 30–90 s | INS drift model |  |
| High-Threat |  |  |  |  | Radar/EW; patrols | Terrain masking |  |
| Multi-Drop |  |  |  |  | 3–5 drops; service 2–8 min | Payload ±15% |  |
| Emergency |  |  |  |  | Abort to nearest LZ | ≤ 2 min commit |  |

## 11. YAML Snippets (Config Templates)

Minimal scenario config (example):

```yaml
scenario:
  name: "urban_short_bangalore"
  start: {lat: 12.9716, lon: 77.5946, alt_m: 120}
  goal:  {lat: 13.0827, lon: 77.5877, alt_m: 120}
  altitude_band_m_agl: [100, 150]
  constraints:
    min_clearance_m: 30
    max_climb_rate_mps: 4.0
    max_descent_rate_mps: 4.0
  toggles:
    roz: false
    dynamic_popups: false
    gps_denied: false

planning:
  grid_resolution_m: 100
  alternatives_k: 3
  smoothing: {enabled: true, curvature_bounded: true}
  cost_weights: {distance: 0.3, energy: 0.3, risk: 0.3, time: 0.1}

robustness:
  enabled: false
```

Robust variant (example):

```yaml
robustness:
  enabled: true
  chance_constraints:
    violation_prob_alpha: 0.05
  uncertainties:
    wind:
      bias_sigma: 0.1
      dryden_length_scale_m: 300
    battery:
      capacity_sigma_fraction: 0.05
    radar:
      snr_threshold_db_jitter: 3
  risk_metrics:
    var_alpha: 0.95
    cvar_alpha: 0.95
  monte_carlo:
    samples: 200
    seeds: [0,1,2,3,4]
```

## 12. Aggregation & Reporting Script (Outline)

- For each run, write `results/{scenario}/{method}/seed_{i}.json` with metrics.
- Aggregate into `reports/{date}/main_results.csv` and `robustness_summary.csv`.
- Generate plots:
  - Pareto fronts per scenario (save as PNG/PDF)
  - Replanning latency histograms
  - Tracking RMS boxplots

This protocol ensures breadth (operational and environmental), depth (robustness and control), and rigor (statistics and reproducibility) suitable for a research paper.
