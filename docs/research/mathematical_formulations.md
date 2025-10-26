# Mathematical Formulations for eVTOL Trajectory Optimization

## 1. Multi-Objective Optimization Problem

### Problem Formulation
```
minimize: F(x) = [f₁(x), f₂(x), f₃(x), f₄(x)]ᵀ
subject to: g(x) ≤ 0, h(x) = 0, x ∈ X
```

Where:
- **f₁(x)**: Risk cost function
- **f₂(x)**: Energy consumption function  
- **f₃(x)**: Flight time function
- **f₄(x)**: Mission success probability (maximized)

### Risk Cost Function
```
f₁(x) = ∫₀ᵀ [wᵣᵢₛₖ · R(x(t), t) + wₜₕᵣₑₐₜ · T(x(t), t)] dt
```

Where:
- R(x(t), t): Terrain risk at position x(t) and time t
- T(x(t), t): Threat probability at position x(t) and time t
- wᵣᵢₛₖ, wₜₕᵣₑₐₜ: Risk and threat weights

### Energy Cost Function
```
f₂(x) = ∫₀ᵀ P(x(t), ẋ(t), t) dt
```

Where P(x(t), ẋ(t), t) is the power consumption function:
```
P(x, ẋ, t) = Pₐₑᵣₒ(x, ẋ) + Pₐₗₜ(x) + Pₐₜₘ(x, t) + Pₐᵤₓ
```

- Pₐₑᵣₒ: Aerodynamic power
- Pₐₗₜ: Altitude-dependent power
- Pₐₜₘ: Atmospheric condition power
- Pₐᵤₓ: Auxiliary systems power

## 2. Perception Fusion Mathematics

### Bayesian Fusion
```
P(risk|x₁, x₂, ..., xₙ) = P(x₁, x₂, ..., xₙ|risk) · P(risk) / P(x₁, x₂, ..., xₙ)
```

### Dempster-Shafer Theory
```
m₁₂(A) = Σ_{B∩C=A} m₁(B) · m₂(C) / (1 - K)
```

Where K is the conflict coefficient:
```
K = Σ_{B∩C=∅} m₁(B) · m₂(C)
```

### Uncertainty Quantification
```
σ²_fused = Σᵢ wᵢ² · σᵢ² + 2Σᵢ<ⱼ wᵢwⱼ · ρᵢⱼ · σᵢσⱼ
```

## 3. Vehicle Dynamics

### 6-DoF Rigid Body Equations
```
m · ẍ = F_aero + F_gravity + F_thrust
I · ω̇ = M_aero + M_thrust + ω × (I · ω)
```

### Battery State Model
```
SOĊ = -I_batt / C_batt
Ṫ_batt = (P_loss - h · A · (T_batt - T_amb)) / (m_batt · c_batt)
```

### Power Consumption Model
```
P_total = P_rotors + P_avionics + P_thermal
P_rotors = Σᵢ (k₁ · ωᵢ³ + k₂ · ωᵢ² + k₃ · ωᵢ)
```

## 4. Robust Optimization

### Chance Constraints
```
P[g(x, ξ) ≤ 0] ≥ 1 - α
```

### Uncertainty Sets
```
U = {ξ : ||ξ - ξ̂||₂ ≤ Γ}
```

### Robust Counterpart
```
minimize: max_{ξ∈U} f(x, ξ)
subject to: g(x, ξ) ≤ 0, ∀ξ ∈ U
```

## 5. Pareto Optimization

### Dominance Relation
```
x₁ ≺ x₂ ⟺ fᵢ(x₁) ≤ fᵢ(x₂) ∀i and fⱼ(x₁) < fⱼ(x₂) for some j
```

### Hypervolume Indicator
```
HV(S) = ∫_{ℝᵐ} I_S(z) dz
```

### NSGA-II Algorithm
1. **Selection**: Tournament selection with crowding distance
2. **Crossover**: Simulated binary crossover (SBX)
3. **Mutation**: Polynomial mutation
4. **Environmental Selection**: Non-dominated sorting + crowding distance

## 6. Computational Complexity

### Time Complexity
- **A* Search**: O(b^d) where b is branching factor, d is depth
- **Pareto Optimization**: O(MN²) where M is objectives, N is population
- **Perception Fusion**: O(n²) where n is number of sensors

### Space Complexity
- **Route Planning**: O(V + E) where V is vertices, E is edges
- **Multi-objective**: O(MN) for Pareto front storage
- **Vehicle Dynamics**: O(1) for state variables

## 7. Convergence Analysis

### Pareto Front Convergence
```
lim_{t→∞} d(P_t, P*) = 0
```

Where:
- P_t is Pareto front at generation t
- P* is true Pareto front
- d(·,·) is Hausdorff distance

### Convergence Rate
```
||x_{k+1} - x*|| ≤ ρ ||x_k - x*||
```

Where ρ < 1 is the convergence rate.

## 8. Performance Bounds

### Approximation Ratio
```
f(ALG) ≤ (1 + ε) · f(OPT)
```

### Competitive Ratio
```
f(ALG) ≤ c · f(OPT)
```

Where c is the competitive ratio.

## 9. Validation Metrics

### Hypervolume Improvement
```
ΔHV = HV(P_new) - HV(P_old)
```

### Inverted Generational Distance (IGD)
```
IGD(P, P*) = (1/|P|) Σ_{p∈P} min_{p*∈P*} d(p, p*)
```

### Spread Metric
```
Δ = (d_f + d_l + Σᵢ|dᵢ - d̄|) / (d_f + d_l + (M-1)d̄)
```

Where:
- d_f, d_l: Distance to extreme points
- dᵢ: Distance between consecutive points
- d̄: Average distance
