"""
Classical Benchmark Functions F1-F23
=====================================
Unimodal (F1-F7), Multimodal (F8-F13), Fixed-Dim Multimodal (F14-F23)
As used in the original SSO paper (Nemati et al., 2024)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class BenchmarkFunction:
    name: str
    dim: int
    lb: float
    ub: float
    optimum: float
    func: Callable
    category: str = ""


def _f1(x):
    return np.sum(x**2)

def _f2(x):
    return np.sum(np.arange(1, len(x)+1) * x**2)

def _f3(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

def _f4(x):
    return np.sum([np.sum(x[:i+1])**2 for i in range(len(x))])

def _f5(x):
    return np.max(np.abs(x))

def _f6(x):
    return np.sum((np.floor(x + 0.5))**2)

def _f7(x):
    d = len(x)
    return np.sum(np.arange(1, d+1) * x**4) + np.random.random()

def _f8(x):
    return np.sum(-x * np.sin(np.sqrt(np.abs(x))))

def _f9(x):
    d = len(x)
    return 10 * d + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def _f10(x):
    d = len(x)
    a = -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / d))
    b = -np.exp(np.sum(np.cos(2 * np.pi * x)) / d)
    return a + b + 20 + np.e

def _f11(x):
    d = len(x)
    return np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, d+1)))) + 1

def _u(x, a, k, m):
    y = np.zeros_like(x, dtype=float)
    y[x > a] = k * (x[x > a] - a) ** m
    y[x < -a] = k * (-x[x < -a] - a) ** m
    return y

def _f12(x):
    d = len(x)
    y = 1 + (x + 1) / 4
    s1 = 10 * np.sin(np.pi * y[0]) ** 2
    s2 = np.sum((y[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * y[1:])**2))
    s3 = (y[-1] - 1)**2
    penalty = np.sum(_u(x, 10, 100, 4))
    return (np.pi / d) * (s1 + s2 + s3) + penalty

def _f13(x):
    d = len(x)
    s1 = np.sin(3 * np.pi * x[0]) ** 2
    s2 = np.sum((x[:-1] - 1)**2 * (1 + np.sin(3 * np.pi * x[1:])**2))
    s3 = (x[-1] - 1)**2 * (1 + np.sin(2 * np.pi * x[-1])**2)
    penalty = np.sum(_u(x, 5, 100, 4))
    return 0.1 * (s1 + s2 + s3) + penalty

def _f14(x):
    aS = np.array([[-32,-16,0,16,32]*5,
                   [-32]*5+[-16]*5+[0]*5+[16]*5+[32]*5], dtype=float)
    bS = np.zeros(25)
    for j in range(25):
        bS[j] = np.sum((x - aS[:, j])**6)
    return (1/500 + np.sum(1 / (np.arange(1, 26) + bS))) ** (-1)

def _f15(x):
    aK = np.array([0.1957,0.1947,0.1735,0.16,0.0844,0.0627,0.0456,0.0342,0.0323,0.0235,0.0246])
    bK = 1 / np.array([0.25,0.5,1,2,4,6,8,10,12,14,16])
    return np.sum((aK - (x[0]*(bK**2+x[1]*bK)) / (bK**2+x[2]*bK+x[3]))**2)

def _f16(x):
    return (4 - 2.1*x[0]**2 + x[0]**4/3)*x[0]**2 + x[0]*x[1] + (-4 + 4*x[1]**2)*x[1]**2

def _f17(x):
    a,b,c,r,s,t = 1,5.1/(4*np.pi**2),5/np.pi,-6,10,1/(8*np.pi)
    return a*(x[1]-b*x[0]**2+c*x[0]-r)**2 + s*(1-t)*np.cos(x[0]) + s

def _f18(x):
    a = 1 + (x[0]+x[1]+1)**2*(19-14*x[0]+3*x[0]**2-14*x[1]+6*x[0]*x[1]+3*x[1]**2)
    b = 30 + (2*x[0]-3*x[1])**2*(18-32*x[0]+12*x[0]**2+48*x[1]-36*x[0]*x[1]+27*x[1]**2)
    return a * b

def _f19(x):
    aH = np.array([[3,0.1,3,0.1],[10,10,10,10],[30,35,30,35]], dtype=float).T
    pH = np.array([[0.3689,0.117,0.2673],[0.4699,0.4387,0.747],[0.1091,0.8732,0.5547],[0.03815,0.5743,0.8828]])
    cH = np.array([1,1.2,3,3.2])
    return -np.sum(cH * np.exp(-np.sum(aH * (x - pH)**2, axis=1)))

def _f20(x):
    aH = np.array([[10,3,17,3.5,1.7,8],[0.05,10,17,0.1,8,14],
                   [3,3.5,1.7,10,17,8],[17,8,0.05,10,0.1,14]])
    pH = np.array([[0.1312,0.1696,0.5569,0.0124,0.8283,0.5886],
                   [0.2329,0.4135,0.8307,0.3736,0.1004,0.9991],
                   [0.2348,0.1415,0.3522,0.2883,0.3047,0.6650],
                   [0.4047,0.8828,0.8732,0.5743,0.1091,0.0381]])
    cH = np.array([1,1.2,3,3.2])
    return -np.sum(cH * np.exp(-np.sum(aH * (x - pH)**2, axis=1)))

def _shekel(x, m):
    aS = np.array([[4,4,4,4],[1,1,1,1],[8,8,8,8],[6,6,6,6],[3,7,3,7],
                   [2,9,2,9],[5,5,3,3],[8,1,8,1],[6,2,6,2],[7,3.6,7,3.6]])
    cS = np.array([0.1,0.2,0.2,0.4,0.4,0.6,0.3,0.7,0.5,0.5])
    return -np.sum([1/(np.sum((x - aS[i])**2) + cS[i]) for i in range(m)])

def _f21(x): return _shekel(x, 5)
def _f22(x): return _shekel(x, 7)
def _f23(x): return _shekel(x, 10)


def get_classical_functions(dim=30):
    """
    Returns all 23 classical benchmark functions.
    dim parameter applies to F1-F13 only; F14-F23 have fixed dimensions.
    """
    functions = [
        BenchmarkFunction("F1-Sphere",         dim, -100, 100,  0,        _f1,  "Unimodal"),
        BenchmarkFunction("F2-SumSquares",      dim,  -10,  10,  0,        _f2,  "Unimodal"),
        BenchmarkFunction("F3-Schwefel2.22",    dim,  -10,  10,  0,        _f3,  "Unimodal"),
        BenchmarkFunction("F4-Schwefel1.2",     dim, -100, 100,  0,        _f4,  "Unimodal"),
        BenchmarkFunction("F5-Schwefel2.21",    dim, -100, 100,  0,        _f5,  "Unimodal"),
        BenchmarkFunction("F6-Step",            dim, -100, 100,  0,        _f6,  "Unimodal"),
        BenchmarkFunction("F7-Quartic",         dim,  -1.28, 1.28, 0,      _f7,  "Unimodal"),
        BenchmarkFunction("F8-Schwefel2.26",    dim, -500, 500, -418.9829*dim, _f8, "Multimodal"),
        BenchmarkFunction("F9-Rastrigin",       dim, -5.12, 5.12, 0,       _f9,  "Multimodal"),
        BenchmarkFunction("F10-Ackley",         dim,  -32,  32,  0,        _f10, "Multimodal"),
        BenchmarkFunction("F11-Griewank",       dim, -600, 600,  0,        _f11, "Multimodal"),
        BenchmarkFunction("F12-Penalized1",     dim,  -50,  50,  0,        _f12, "Multimodal"),
        BenchmarkFunction("F13-Penalized2",     dim,  -50,  50,  0,        _f13, "Multimodal"),
        BenchmarkFunction("F14-ShekelFoxholes", 2,  -65.5, 65.5, 1,       _f14, "Fixed-Dim"),
        BenchmarkFunction("F15-Kowalik",        4,   -5,   5,  0.0003,    _f15, "Fixed-Dim"),
        BenchmarkFunction("F16-SixHumpCamel",   2,   -5,   5, -1.0316,    _f16, "Fixed-Dim"),
        BenchmarkFunction("F17-Branin",         2,   -5,  15,  0.398,     _f17, "Fixed-Dim"),
        BenchmarkFunction("F18-GoldsteinPrice", 2,   -2,   2,  3,         _f18, "Fixed-Dim"),
        BenchmarkFunction("F19-Hartmann3",      3,    0,   1, -3.86,      _f19, "Fixed-Dim"),
        BenchmarkFunction("F20-Hartmann6",      6,    0,   1, -3.3223,    _f20, "Fixed-Dim"),
        BenchmarkFunction("F21-Shekel5",        4,    0,  10, -10.1532,   _f21, "Fixed-Dim"),
        BenchmarkFunction("F22-Shekel7",        4,    0,  10, -10.4029,   _f22, "Fixed-Dim"),
        BenchmarkFunction("F23-Shekel10",       4,    0,  10, -10.5364,   _f23, "Fixed-Dim"),
    ]
    return functions
