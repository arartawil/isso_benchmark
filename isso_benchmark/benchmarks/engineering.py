"""
Engineering Design Optimization Problems
==========================================
Constrained real-world problems commonly used to validate metaheuristics.
Constraint handling: exterior penalty method.

Reference:
  Coello, C. A. C. (2002). Theoretical and numerical constraint-handling
  techniques used with evolutionary algorithms: a survey of the state of the art.

Problem 1: Welded Beam Design
  - 4 design variables (h, l, t, b)
  - 7 inequality constraints
  - Known best: f* ≈ 1.724852
  - Source: Ragsdell & Phillips (1976)

Problem 2: Pressure Vessel Design
  - 4 design variables (Ts, Th, R, L)
  - 4 inequality constraints
  - Known best: f* ≈ 6059.7143
  - Source: Kannan & Kramer (1994)

Problem 3: Tension/Compression Spring Design
  - 3 design variables (d, D, N)
  - 4 inequality constraints
  - Known best: f* ≈ 0.012665
  - Source: Belegundu (1982); Arora (1989)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Union


@dataclass
class BenchmarkFunction:
    """Same structure as classical.py for compatibility with Runner."""
    name: str
    dim: int
    lb: object        # float or np.ndarray
    ub: object        # float or np.ndarray
    optimum: float
    func: Callable
    category: str = ""


# ======================== Constants ========================

_P      = 6000.0        # Applied load (lb)
_L      = 14.0          # Beam length (in)
_E      = 30e6          # Young's modulus (psi)
_G      = 12e6          # Shear modulus (psi)
_tau_max = 13600.0       # Max shear stress (psi)
_sigma_max = 30000.0     # Max normal stress (psi)
_delta_max = 0.25        # Max deflection (in)

# Penalty coefficient — large enough to push infeasible solutions away
_PENALTY = 1e6


# ======================== Welded Beam ========================

def _welded_beam_raw(x):
    """Objective: minimise fabrication cost."""
    h, l, t, b = x[0], x[1], x[2], x[3]
    return 1.10471 * h**2 * l + 0.04811 * t * b * (14.0 + l)


def _welded_beam_constraints(x):
    """
    Return a list of constraint violations g_i(x).
    Convention: g_i(x) <= 0  means feasible.
    """
    h, l, t, b = x[0], x[1], x[2], x[3]

    # ---------- Shear stress τ ----------
    tau_prime = _P / (np.sqrt(2) * h * l)
    M = _P * (_L + l / 2.0)
    R = np.sqrt(l**2 / 4.0 + ((h + t) / 2.0)**2)
    J = 2.0 * (np.sqrt(2) * h * l * (l**2 / 12.0 + ((h + t) / 2.0)**2))
    tau_double = M * R / J
    tau = np.sqrt(tau_prime**2
                  + 2.0 * tau_prime * tau_double * l / (2.0 * R)
                  + tau_double**2)

    # ---------- Bending stress σ ----------
    sigma = 6.0 * _P * _L / (b * t**2)

    # ---------- Deflection δ ----------
    delta = 4.0 * _P * _L**3 / (_E * t**3 * b)

    # ---------- Buckling load P_c ----------
    Pc = (4.013 * _E * np.sqrt(t**2 * b**6 / 36.0) / _L**2
          * (1.0 - (t / (2.0 * _L)) * np.sqrt(_E / (4.0 * _G))))

    # Constraint list: g_i <= 0 → feasible
    g1 = tau - _tau_max              # τ ≤ τ_max
    g2 = sigma - _sigma_max          # σ ≤ σ_max
    g3 = h - b                       # h ≤ b
    g4 = delta - _delta_max          # δ ≤ δ_max
    g5 = _P - Pc                     # P ≤ P_c
    g6 = 0.125 - h                   # 0.125 ≤ h
    g7 = (1.10471 * h**2 * l
          + 0.04811 * t * b * (14.0 + l)) - 5.0  # cost ≤ 5

    return [g1, g2, g3, g4, g5, g6, g7]


def _penalty_sum(constraints):
    """Quadratic penalty for violated constraints."""
    total = 0.0
    for g in constraints:
        if g > 0:
            total += g**2
    return total


def welded_beam_penalised(x):
    """
    Penalised objective for the Welded Beam Design problem.
    f_pen(x) = f(x) + λ * Σ max(0, g_i)²
    """
    f = _welded_beam_raw(x)
    violations = _welded_beam_constraints(x)
    return f + _PENALTY * _penalty_sum(violations)


# ======================== Pressure Vessel Design ========================
# Variables: x1 = Ts (shell thickness), x2 = Th (head thickness),
#            x3 = R  (inner radius),    x4 = L  (length without heads)
# Objective: minimise total cost (material + forming + welding)
# 4 inequality constraints
# Known optimum: f* ≈ 6059.7143  at x* = (0.8125, 0.4375, 42.0984, 176.6366)

def _pressure_vessel_raw(x):
    """Objective: minimise total cost."""
    Ts, Th, R, L = x[0], x[1], x[2], x[3]
    return (0.6224 * Ts * R * L
            + 1.7781 * Th * R**2
            + 3.1661 * Ts**2 * L
            + 19.84 * Ts**2 * R)


def _pressure_vessel_constraints(x):
    """4 inequality constraints, g_i <= 0 means feasible."""
    Ts, Th, R, L = x[0], x[1], x[2], x[3]

    g1 = -Ts + 0.0193 * R                           # Ts >= 0.0193*R
    g2 = -Th + 0.00954 * R                           # Th >= 0.00954*R
    g3 = (-np.pi * R**2 * L
           - (4.0 / 3.0) * np.pi * R**3
           + 1296000.0)                               # Volume >= 1,296,000 in³
    g4 = L - 240.0                                    # L <= 240

    return [g1, g2, g3, g4]


def pressure_vessel_penalised(x):
    """Penalised objective for Pressure Vessel Design."""
    f = _pressure_vessel_raw(x)
    violations = _pressure_vessel_constraints(x)
    return f + _PENALTY * _penalty_sum(violations)


# ======================== Tension/Compression Spring ========================
# Variables: x1 = d (wire diameter), x2 = D (mean coil diameter),
#            x3 = N (number of active coils)
# Objective: minimise spring weight
# 4 inequality constraints
# Known optimum: f* ≈ 0.012665  at x* = (0.05169, 0.35674, 11.2885)

def _spring_raw(x):
    """Objective: minimise spring weight."""
    d, D, N = x[0], x[1], x[2]
    return (N + 2.0) * D * d**2


def _spring_constraints(x):
    """4 inequality constraints, g_i <= 0 means feasible."""
    d, D, N = x[0], x[1], x[2]

    g1 = 1.0 - (D**3 * N) / (71785.0 * d**4)        # Shear stress
    g2 = ((4.0 * D**2 - d * D) / (12566.0 * (D * d**3 - d**4))
          + 1.0 / (5108.0 * d**2) - 1.0)             # Surge frequency
    g3 = 1.0 - 140.45 * d / (D**2 * N)               # Deflection
    g4 = (d + D) / 1.5 - 1.0                         # Outer diameter

    return [g1, g2, g3, g4]


def spring_penalised(x):
    """Penalised objective for Tension/Compression Spring Design."""
    f = _spring_raw(x)
    violations = _spring_constraints(x)
    return f + _PENALTY * _penalty_sum(violations)


# ======================== Public API ========================

def get_engineering_functions():
    """
    Returns a list of engineering design benchmark problems.
    Each entry is compatible with the Runner class.

    Includes:
      E1 - Welded Beam Design            (4-D, 7 constraints)
      E2 - Pressure Vessel Design        (4-D, 4 constraints)
      E3 - Tension/Compression Spring    (3-D, 4 constraints)
    """
    functions = [
        BenchmarkFunction(
            name="E1-WeldedBeam",
            dim=4,
            lb=np.array([0.1,   0.1, 0.1, 0.1]),
            ub=np.array([2.0,  10.0, 10.0, 2.0]),
            optimum=1.724852,
            func=welded_beam_penalised,
            category="Engineering",
        ),
        BenchmarkFunction(
            name="E2-PressureVessel",
            dim=4,
            lb=np.array([0.0,    0.0,   10.0,  10.0]),
            ub=np.array([99.0,  99.0,  200.0, 200.0]),
            optimum=6059.7143,
            func=pressure_vessel_penalised,
            category="Engineering",
        ),
        BenchmarkFunction(
            name="E3-Spring",
            dim=3,
            lb=np.array([0.05, 0.25, 2.0]),
            ub=np.array([2.00, 1.30, 15.0]),
            optimum=0.012665,
            func=spring_penalised,
            category="Engineering",
        ),
    ]
    return functions


# ======================== Standalone check ========================

if __name__ == "__main__":
    print("=" * 60)
    print("  Engineering Problems — Known Optima Sanity Check")
    print("=" * 60)

    # E1: Welded Beam
    x1 = np.array([0.205730, 3.470489, 9.036624, 0.205729])
    print(f"\nE1-WeldedBeam  x* = {x1}")
    print(f"  Raw cost   = {_welded_beam_raw(x1):.6f}")
    print(f"  Penalised  = {welded_beam_penalised(x1):.6f}")
    print(f"  Constraints= {[f'{g:.4f}' for g in _welded_beam_constraints(x1)]}")

    # E2: Pressure Vessel
    x2 = np.array([0.8125, 0.4375, 42.0984, 176.6366])
    print(f"\nE2-PressureVessel  x* = {x2}")
    print(f"  Raw cost   = {_pressure_vessel_raw(x2):.4f}")
    print(f"  Penalised  = {pressure_vessel_penalised(x2):.4f}")
    print(f"  Constraints= {[f'{g:.4f}' for g in _pressure_vessel_constraints(x2)]}")

    # E3: Spring
    x3 = np.array([0.05169, 0.35674, 11.2885])
    print(f"\nE3-Spring  x* = {x3}")
    print(f"  Raw cost   = {_spring_raw(x3):.6f}")
    print(f"  Penalised  = {spring_penalised(x3):.6f}")
    print(f"  Constraints= {[f'{g:.6f}' for g in _spring_constraints(x3)]}")
