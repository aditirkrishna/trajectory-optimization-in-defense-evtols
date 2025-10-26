# Algorithms & Data Structures Documentation

## Complete Technical Reference for eVTOL Trajectory Optimization

---

## Table of Contents
1. [Pathfinding Algorithms](#pathfinding-algorithms)
2. [Optimization Algorithms](#optimization-algorithms)
3. [Control Algorithms](#control-algorithms)
4. [Data Structures](#data-structures)
5. [Mathematical Models](#mathematical-models)

---

## 1. Pathfinding Algorithms

### 1.1 A* (A-Star) Algorithm
**Location**: `planning-layer/src/planning_layer/routing/planner.py`

**Purpose**: Find optimal path from start to goal considering multiple objectives

**Algorithm Overview**:
```
A* uses f(n) = g(n) + h(n) where:
- g(n) = actual cost from start to node n
- h(n) = heuristic estimate from n to goal
- f(n) = total estimated cost through n
```

**Implementation Details**:
- **Data Structure**: Min-heap (priority queue) for open set
- **Heuristic**: Haversine distance (admissible)
- **Cost Function**: Weighted sum of distance, energy, and risk
- **Time Complexity**: O((V+E) log V) where V=nodes, E=edges
- **Space Complexity**: O(V)

**Key Features**:
1. **Multi-objective cost**:
   ```python
   cost = w_dist * distance + w_energy * energy + w_risk * risk
   ```
2. **Early termination** when goal found
3. **Closed set** to avoid revisiting nodes
4. **3D search space** (lat, lon, altitude)

**Optimizations**:
- Grid-based discretization for manageable state space
- Lazy evaluation of neighbors
- Efficient hash-based closed set

---

### 1.2 Graph-Based Routing
**Location**: `planning-layer/src/planning_layer/routing/graph_router.py`

**Purpose**: Find k-best alternative routes using graph algorithms

**Algorithm**: Modified Yen's K-Shortest Paths Algorithm

**How It Works**:
1. Find shortest path using A*
2. For each deviation point:
   - Remove edge from previous path
   - Find new shortest path
   - Add to candidate list
3. Sort candidates and select k-best

**Time Complexity**: O(k * n * (m + n log n))
- k = number of paths
- n = nodes, m = edges

**Use Case**: Provide diverse route options with different trade-offs

---

## 2. Optimization Algorithms

### 2.1 Pareto Frontier Optimization
**Location**: `planning-layer/src/planning_layer/optimization/pareto.py`

**Purpose**: Find non-dominated solutions in multi-objective space

**Core Concept**:
```
Solution A dominates B if:
- A is better in at least one objective
- A is not worse in any objective
```

**Algorithm**:
```python
def find_pareto_front(solutions):
    front = []
    for sol in solutions:
        dominated = False
        to_remove = []
        
        for existing in front:
            if existing.dominates(sol):
                dominated = True
                break
            if sol.dominates(existing):
                to_remove.append(existing)
        
        if not dominated:
            front = [s for s in front if s not in to_remove]
            front.append(sol)
    
    return front
```

**Time Complexity**: O(n² * m) where n=solutions, m=objectives

**Key Features**:
1. **Knee Point Detection**: Find best trade-off solution
2. **Diversity Selection**: Select k diverse solutions
3. **Scalarization Methods**: 
   - Weighted sum
   - Tchebycheff

---

### 2.2 Robust Optimization
**Location**: `planning-layer/src/planning_layer/robust/uncertainty_planning.py`

**Purpose**: Plan routes that remain feasible under uncertainty

**Methods**:

#### 2.2.1 Chance-Constrained Optimization
```
P(cost ≤ threshold) ≥ α

For normal distribution:
μ + z_α * σ ≤ threshold
```

**Implementation**:
```python
def chance_constraint(mean, std, threshold, confidence):
    z_score = stats.norm.ppf(confidence)
    upper_bound = mean + z_score * std
    return upper_bound <= threshold
```

#### 2.2.2 Value at Risk (VaR)
```
VaR_α = α-percentile of cost distribution
```

#### 2.2.3 Conditional Value at Risk (CVaR)
```
CVaR_α = E[cost | cost ≥ VaR_α]
```

**Mathematical Foundation**:
- Assumes Gaussian uncertainty (can be extended)
- Conservative estimates via confidence intervals
- Monte Carlo validation with 1000+ samples

---

### 2.3 Trajectory Smoothing
**Location**: `planning-layer/src/planning_layer/smoothing/spline_smoother.py`

**Purpose**: Generate smooth, flyable trajectories from waypoints

**Algorithm**: Curvature-Bounded B-Splines

**Mathematical Model**:
```
B-spline: S(u) = Σ P_i * B_{i,k}(u)

where:
- P_i = control points
- B_{i,k} = basis functions of degree k
- u ∈ [0,1] = parameter
```

**Curvature Constraint**:
```
κ(s) = ||r''(s)|| / ||r'(s)||³ ≤ κ_max

where κ_max = 1/min_turn_radius
```

**Implementation Steps**:
1. Fit cubic B-spline through waypoints
2. Compute curvature at sample points
3. If κ > κ_max, adjust control points
4. Iterate until constraints satisfied

**Time Complexity**: O(n * m) where n=waypoints, m=samples

---

### 2.4 Energy Optimization
**Location**: `planning-layer/src/planning_layer/energy/optimizer.py`

**Purpose**: Minimize energy consumption along route

**Energy Model**:
```
E_total = Σ (d_i * e_i)

where:
- d_i = segment distance
- e_i = energy cost per km (from perception)
```

**Factors Considered**:
- Terrain elevation changes
- Wind conditions
- Altitude profile
- Speed profile

**Optimization Strategy**:
- Greedy selection of low-energy segments
- Reserve battery constraint (typically 15-20%)
- Real-time energy estimation

---

## 3. Control Algorithms

### 3.1 PID Control
**Location**: `control-layer/src/control/flight_controller.py`

**Purpose**: Track desired trajectory with position/velocity/attitude control

**Mathematical Model**:
```
u(t) = K_p * e(t) + K_i * ∫e(τ)dτ + K_d * de/dt

where:
- e(t) = error signal
- K_p, K_i, K_d = tuning gains
```

**Hierarchical Architecture**:
```
Position Controller (Outer Loop)
    ↓
Velocity Controller (Middle Loop)
    ↓
Attitude Controller (Inner Loop)
```

**Tuning Guidelines**:
- Position: K_p=2.0, K_i=0.1, K_d=1.0
- Velocity: K_p=1.5, K_i=0.2, K_d=0.5
- Attitude: K_p=3.0, K_i=0.1, K_d=0.8

**Anti-Windup**: Limit integral term to ±10.0

---

### 3.2 Trajectory Generation
**Location**: `control-layer/src/control/trajectory_generator.py`

**Purpose**: Generate time-parameterized, dynamically feasible trajectories

**Algorithm**: Cubic Hermite Spline Interpolation

**Mathematical Basis**:
```
p(τ) = h₁(τ)*p₀ + h₂(τ)*v₀ + h₃(τ)*p₁ + h₄(τ)*v₁

where:
- h_i(τ) = Hermite basis functions
- τ ∈ [0,1] = normalized time
- p = position, v = velocity
```

**Hermite Basis Functions**:
```
h₁(τ) = 2τ³ - 3τ² + 1
h₂(τ) = τ³ - 2τ² + τ
h₃(τ) = -2τ³ + 3τ²
h₄(τ) = τ³ - τ²
```

**Properties**:
- C² continuous (smooth acceleration)
- Exact interpolation at waypoints
- Velocity control at endpoints

---

## 4. Data Structures

### 4.1 Priority Queue (Min-Heap)
**Used In**: A* algorithm

**Operations**:
- Insert: O(log n)
- Extract-Min: O(log n)
- Decrease-Key: O(log n)

**Implementation**: Python `heapq` module

---

### 4.2 Hash Set
**Used In**: A* closed set, visited nodes

**Operations**:
- Insert: O(1) average
- Lookup: O(1) average
- Delete: O(1) average

**Implementation**: Python `set` with custom `__hash__`

---

### 4.3 Grid Graph
**Used In**: Discretized 3D space

**Structure**:
```
Grid[i][j][k] = Node(lat, lon, alt)

Connectivity: 26-connected (3D)
- 6 face neighbors
- 12 edge neighbors
- 8 corner neighbors
```

**Space Optimization**: 
- Sparse representation
- Only store visited nodes

---

### 4.4 Quadtree/Octree
**Used In**: Spatial indexing for fast queries

**Properties**:
- Hierarchical space partitioning
- Query: O(log n) average
- Good for non-uniform data

---

## 5. Mathematical Models

### 5.1 Haversine Distance
**Purpose**: Calculate great-circle distance on Earth

**Formula**:
```
a = sin²(Δφ/2) + cos(φ₁)*cos(φ₂)*sin²(Δλ/2)
c = 2*atan2(√a, √(1-a))
d = R * c

where:
- φ = latitude
- λ = longitude
- R = Earth radius (6371 km)
```

**Accuracy**: ±0.5% for distances < 1000 km

---

### 5.2 Wind Effect Model
**Location**: `perception-layer/src/atmosphere/wind_model.py`

**Purpose**: Model wind impact on flight

**Interpolation Methods**:

#### Linear Interpolation:
```
f(x,y,z) = Σ w_i * f_i

where w_i = interpolation weights
```

#### RBF (Radial Basis Function):
```
f(x) = Σ λ_i * φ(||x - x_i||)

where φ(r) = exp(-εr²) (Gaussian RBF)
```

#### IDW (Inverse Distance Weighting):
```
f(x) = Σ (w_i * f_i) / Σ w_i

where w_i = 1 / d_i^p (typically p=2)
```

---

### 5.3 Turbulence Model
**Location**: `perception-layer/src/atmosphere/turbulence_model.py`

**Method**: Dryden Spectral Model

**Mathematical Basis**:
```
Turbulence PSD: Φ(ω) = σ² * L / (π * (1 + (Lω)²))

where:
- σ = turbulence intensity
- L = length scale
- ω = spatial frequency
```

**Implementation**: Time-series generation via spectral factorization

---

### 5.4 Radar Detection Model
**Location**: `perception-layer/src/threats/radar_model.py`

**Radar Equation**:
```
SNR = (P_t * G² * λ² * σ) / ((4π)³ * R⁴ * k * T * B * L)

where:
- P_t = transmit power
- G = antenna gain
- λ = wavelength
- σ = radar cross-section
- R = range
- k, T, B, L = system parameters
```

**Detection Probability**:
```
P_d = 1 / (1 + exp(-k*(SNR - SNR_threshold)))

Logistic function with threshold
```

---

### 5.5 Battery Model
**Location**: `vehicle-layer/src/energy/battery_model.py`

**State of Charge (SOC) Dynamics**:
```
dSOC/dt = -I / C

where:
- I = current draw
- C = battery capacity
```

**Voltage Model**:
```
V = V_oc - I*R_internal

where:
- V_oc = open circuit voltage (function of SOC)
- R = internal resistance
```

**Temperature Dynamics**:
```
dT/dt = (P_loss - P_cooling) / (m * c_p)

where:
- P_loss = I² * R (Joule heating)
- P_cooling = h * A * (T - T_amb)
```

---

## 6. Complexity Analysis Summary

| Algorithm | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| A* Pathfinding | O((V+E) log V) | O(V) |
| K-Shortest Paths | O(k*n*(m + n log n)) | O(kn) |
| Pareto Frontier | O(n²*m) | O(n) |
| B-Spline Smoothing | O(n*m) | O(n) |
| PID Control | O(1) per step | O(1) |
| Trajectory Generation | O(n) | O(n) |
| Wind Interpolation (RBF) | O(n³) setup, O(n) query | O(n²) |
| Monte Carlo Validation | O(k*n) | O(k) |

---

## 7. Performance Optimizations

### 7.1 Implemented Optimizations
1. **Grid Discretization**: Reduces state space from infinite to manageable
2. **Lazy Evaluation**: Only compute neighbors when needed
3. **Early Termination**: Stop when goal found (A*)
4. **Caching**: Store computed costs
5. **Vectorization**: Use NumPy for array operations
6. **Sparse Representations**: Only store non-zero/visited elements

### 7.2 Parallelization Opportunities
- Multi-threaded k-best path search
- Parallel Monte Carlo sampling
- GPU-accelerated trajectory smoothing

---

## 8. References & Theory

### Academic Foundations
1. **A* Algorithm**: Hart, P. E., Nilsson, N. J., & Raphael, B. (1968)
2. **Pareto Optimization**: Miettinen, K. (1999). Nonlinear Multiobjective Optimization
3. **PID Control**: Åström, K. J., & Hägglund, T. (2006). Advanced PID Control
4. **B-Splines**: De Boor, C. (2001). A Practical Guide to Splines
5. **Robust Optimization**: Ben-Tal, A., El Ghaoui, L., & Nemirovski, A. (2009)

### Industry Standards
- **Haversine Formula**: Aviation standard for distance calculation
- **Dryden Turbulence**: US Military Specification MIL-F-8785C
- **Radar Equation**: IEEE Standard for Radar Systems

---

## 9. Algorithm Selection Guide

**Choose A* when**:
- Single optimal path needed
- Admissible heuristic available
- Memory not constrained

**Choose K-Shortest Paths when**:
- Multiple alternative routes needed
- Diverse solutions desired
- Risk mitigation important

**Choose Pareto Frontier when**:
- Multiple conflicting objectives
- Trade-off analysis needed
- Decision support required

**Choose Robust Optimization when**:
- High uncertainty
- Safety-critical application
- Conservative estimates needed

---

**Document Version**: 1.0  
**Last Updated**: October 12, 2024  
**Maintained By**: eVTOL Project Team

