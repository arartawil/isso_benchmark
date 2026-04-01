"""
CEC-BC-2017 Benchmark Functions
================================
Shifted and rotated functions used in the SSO paper.
Functions: F1, F3-F13 (F2 excluded - ill-conditioned in high-D)
Dimensions tested: 10, 30, 50, 100
"""
import numpy as np
from dataclasses import dataclass
from typing import Callable


@dataclass
class CEC2017Function:
    name: str
    func_id: int
    dim: int
    lb: float
    ub: float
    optimum: float
    func: Callable
    category: str = ""


def _make_shift_rotate(func_id, dim, seed_base=2017):
    """Generate fixed shift vector and rotation matrix for a given function and dimension."""
    rng = np.random.default_rng(seed_base * 100 + func_id * 10 + dim)
    shift = rng.uniform(-80, 80, dim)
    # Rotation matrix via QR decomposition
    A = rng.standard_normal((dim, dim))
    Q, _ = np.linalg.qr(A)
    return shift, Q


def _bent_cigar(z):
    return z[0]**2 + 1e6 * np.sum(z[1:]**2)

def _zakharov(z):
    d = len(z)
    s1 = np.sum(z**2)
    s2 = np.sum(0.5 * np.arange(1, d+1) * z)
    return s1 + s2**2 + s2**4

def _rosenbrock(z):
    z = z + 1
    return np.sum(100*(z[1:] - z[:-1]**2)**2 + (z[:-1]-1)**2)

def _rastrigin(z):
    return np.sum(z**2 - 10*np.cos(2*np.pi*z) + 10)

def _scaffer_f6(z):
    d = len(z)
    total = 0
    for i in range(d-1):
        xi, xi1 = z[i], z[i+1]
        t = xi**2 + xi1**2
        total += t**0.25 * (np.sin(50*t**0.1)**2 + 1)
    return total

def _lunacek_bi_rastrigin(z):
    d = len(z)
    mu0, s, d_val = 2.5, 1 - 1/(2*np.sqrt(d+20)-8.2), 1
    mu1 = -np.sqrt((mu0**2 - d_val) / s)
    s1 = np.sum((z - mu0)**2)
    s2 = d_val*d + s*np.sum((z - mu1)**2)
    s3 = np.sum(1 - np.cos(2*np.pi*(z - mu0)))
    return min(s1, s2) + 10*s3

def _nc_rastrigin(z):
    d = len(z)
    y = np.round(2*z) / 2
    mask = np.abs(z - y) >= 0.5
    y[mask] = np.floor(2*z[mask] + 0.5) / 2
    return np.sum(y**2 - 10*np.cos(2*np.pi*y) + 10)

def _levy(z):
    d = len(z)
    w = 1 + (z - 1) / 4
    t1 = np.sin(np.pi * w[0])**2
    t2 = np.sum((w[:-1]-1)**2 * (1 + 10*np.sin(np.pi*w[1:])**2))
    t3 = (w[-1]-1)**2 * (1 + np.sin(2*np.pi*w[-1])**2)
    return t1 + t2 + t3

def _hybrid1(z):
    d = len(z)
    p = [0.2, 0.4, 0.4]
    n1, n2 = int(np.ceil(p[0]*d)), int(np.ceil(p[1]*d))
    z1, z2, z3 = z[:n1], z[n1:n1+n2], z[n1+n2:]
    return _zakharov(z1) + _rosenbrock(z2) + _rastrigin(z3)

def _hybrid2(z):
    d = len(z)
    p = [0.3, 0.3, 0.4]
    n1, n2 = int(np.ceil(p[0]*d)), int(np.ceil(p[1]*d))
    z1, z2, z3 = z[:n1], z[n1:n1+n2], z[n1+n2:]
    return _rastrigin(z1) + _scaffer_f6(z2) + _levy(z3)

def _composition1(z):
    d = len(z)
    sigma = [10, 20, 30]
    delta = [1, 10, 1e4]
    funcs = [_rosenbrock, _rastrigin, _levy]
    w, f_vals = [], []
    for i in range(3):
        w.append(np.exp(-np.sum(z**2) / (2 * d * sigma[i]**2)))
        f_vals.append(delta[i] * funcs[i](z))
    w = np.array(w)
    f_vals = np.array(f_vals)
    if np.sum(w) == 0:
        w = np.ones(3)
    return np.sum(w * f_vals) / np.sum(w)

def _composition2(z):
    d = len(z)
    sigma = [20, 20, 30]
    delta = [1, 10, 1e4]
    funcs = [_rastrigin, _scaffer_f6, _levy]
    w, f_vals = [], []
    for i in range(3):
        w.append(np.exp(-np.sum(z**2) / (2 * d * sigma[i]**2)))
        f_vals.append(delta[i] * funcs[i](z))
    w = np.array(w)
    f_vals = np.array(f_vals)
    if np.sum(w) == 0:
        w = np.ones(3)
    return np.sum(w * f_vals) / np.sum(w)


# Map func_id to base function
_BASE_FUNCS = {
    1:  (_bent_cigar,           "Unimodal"),
    3:  (_zakharov,             "Unimodal"),
    4:  (_rosenbrock,           "Multimodal"),
    5:  (_rastrigin,            "Multimodal"),
    6:  (_scaffer_f6,           "Multimodal"),
    7:  (_lunacek_bi_rastrigin, "Multimodal"),
    8:  (_nc_rastrigin,         "Multimodal"),
    9:  (_levy,                 "Multimodal"),
    10: (_hybrid1,              "Hybrid"),
    11: (_hybrid2,              "Hybrid"),
    12: (_composition1,         "Composition"),
    13: (_composition2,         "Composition"),
}

_FUNC_NAMES = {
    1:  "F1-BentCigar",
    3:  "F3-Zakharov",
    4:  "F4-Rosenbrock",
    5:  "F5-Rastrigin",
    6:  "F6-ScafferF6",
    7:  "F7-LunacekBiRastrigin",
    8:  "F8-NC-Rastrigin",
    9:  "F9-Levy",
    10: "F10-Hybrid1",
    11: "F11-Hybrid2",
    12: "F12-Composition1",
    13: "F13-Composition2",
}


def _make_cec_func(func_id, dim):
    """Create shifted+rotated CEC-2017 function for given func_id and dim."""
    base_func, category = _BASE_FUNCS[func_id]
    shift, M = _make_shift_rotate(func_id, dim)

    def cec_func(x):
        z = M @ (x - shift)
        return base_func(z)

    return cec_func, category


def get_cec2017_functions(dims=None):
    """
    Returns CEC-BC-2017 functions for specified dimensions.
    
    Parameters
    ----------
    dims : list of int, optional
        Dimensions to include. Default: [10, 30, 50, 100]
    
    Returns
    -------
    list of CEC2017Function
    """
    if dims is None:
        dims = [10, 30, 50, 100]

    func_ids = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    functions = []

    for dim in dims:
        for fid in func_ids:
            cec_func, category = _make_cec_func(fid, dim)
            functions.append(CEC2017Function(
                name=f"{_FUNC_NAMES[fid]}-D{dim}",
                func_id=fid,
                dim=dim,
                lb=-100.0,
                ub=100.0,
                optimum=fid * 100.0,
                func=cec_func,
                category=category,
            ))

    return functions
